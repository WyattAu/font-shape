//! Glyph rasterization: from a [`font_model`] outline to a coverage bitmap.
//!
//! A glyph arrives in font units, y **up**, on a baseline. This module converts
//! it to device pixels, y **down**, and fills it with the scanline rasteriser.
//! Everything between the two lives in **26.6 fixed point** — 26 integer bits
//! and 6 fractional bits, so one pixel is exactly `64`:
//!
//! ```
//! use font_shape::{from_fixed, to_fixed};
//!
//! assert_eq!(font_shape::PIXEL, 64);
//! assert_eq!(to_fixed(1.0), 64);            // one pixel
//! assert_eq!(to_fixed(0.5), 32);            // half a pixel
//! assert_eq!(from_fixed(96), 1.5);          // and back
//! ```
//!
//! Fixed point is not decoration: it is what makes a glyph land on the same
//! pixel grid at 12 pt and at 144 pt, and what makes
//! [`Hinting::GridFit`] mean "snap to the pixel grid" rather than "round a
//! float and hope".
//!
//! # Scale
//!
//! The scale from font units to 26.6 units is
//! `size_px · 64 / units_per_em`, from [`fixed_scale`]. Everything else in
//! this module is a multiply by that number.
//!
//! ```
//! use font_shape::fixed_scale;
//!
//! // A 1000-units-per-em font at 16 px: one font unit is 16/1000 px.
//! let s = fixed_scale(16.0, 1000).unwrap();
//! assert!((s - 16.0 * 64.0 / 1000.0).abs() < 1e-4, "{s}");
//!
//! // A zero units-per-em font has no scale at all.
//! assert!(fixed_scale(16.0, 0).is_err());
//! ```

use alloc::vec::Vec;

use font_model::{Font, FontMetrics, GlyphId, Outline};

use crate::affine::Affine2D;
use crate::error::ShapeError;
use crate::path::{Path2D, PathBuilder};
use crate::raster::{fill_path, mask_coverage, FillRule, MAX_MASK_ELEMENTS};

/// The number of fractional bits in a 26.6 fixed-point value.
pub const PIXEL_FP_BITS: u32 = 6;

/// One pixel, in 26.6 fixed point.
pub const PIXEL: i32 = 1 << PIXEL_FP_BITS;

/// How a glyph's outline is snapped to the pixel grid.
///
/// Hinting is a *discretion*, not a default: unhinted rendering is what a
/// modern rasteriser wants, and grid fitting is what reproduces a bitmap-era
/// look. Both are offered, and neither is applied behind the caller's back.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Hinting {
    /// Leave the outline where the scale puts it. Every edge gets
    /// antialiased coverage, and a stem that should be 1.3 px wide is 1.3 px
    /// wide. The default.
    #[default]
    None,
    /// Snap every outline point — control points included — to the nearest
    /// pixel boundary, and round advances and bearings to whole pixels.
    ///
    /// This is a coarse integer-grid fit, not TrueType's interpreter: it snaps
    /// points, it does not walk instructions. What it buys is crisp stems and
    /// integer advances at the small sizes where hinting matters.
    GridFit,
}

impl Hinting {
    /// The opposite mode, for a toggle.
    #[must_use]
    pub const fn toggled(self) -> Hinting {
        match self {
            Hinting::None => Hinting::GridFit,
            Hinting::GridFit => Hinting::None,
        }
    }

    /// True for [`Hinting::GridFit`].
    #[must_use]
    pub const fn is_grid_fit(self) -> bool {
        matches!(self, Hinting::GridFit)
    }
}

/// `value` as a 26.6 fixed-point integer, rounded to nearest and saturating.
///
/// ```
/// use font_shape::to_fixed;
///
/// assert_eq!(to_fixed(0.0), 0);
/// assert_eq!(to_fixed(10.0), 640);
/// assert_eq!(to_fixed(-0.25), -16);
/// // Non-finite input cannot be represented; it becomes zero rather than a
/// // wild number that would blow up every later multiply.
/// assert_eq!(to_fixed(f32::NAN), 0);
/// assert_eq!(to_fixed(f32::INFINITY), i32::MAX);
/// ```
#[must_use]
pub fn to_fixed(value: f32) -> i32 {
    if value.is_nan() {
        return 0;
    }
    let scaled = libm::roundf(value * PIXEL as f32);
    // Saturate rather than wrap: a glyph at 1e9 px is off-screen, not an
    // integer-overflow bug report.
    if scaled >= i32::MAX as f32 {
        return i32::MAX;
    }
    if scaled <= i32::MIN as f32 {
        return i32::MIN;
    }
    scaled as i32
}

/// A value that is *already* in 26.6 units, rounded to the nearest integer.
///
/// [`to_fixed`] takes pixels and scales them up; this one takes 26.6 units and
/// only rounds. [`fixed_scale`] returns a 26.6-per-font-unit factor, so this
/// is what a scale multiplication must use.
#[must_use]
pub fn to_26_6(value: f32) -> i32 {
    to_fixed(value / PIXEL as f32)
}

/// A 26.6 fixed-point integer back to pixels.
///
/// ```
/// use font_shape::from_fixed;
///
/// assert_eq!(from_fixed(0), 0.0);
/// assert_eq!(from_fixed(64), 1.0);
/// assert_eq!(from_fixed(-32), -0.5);
/// ```
#[must_use]
pub fn from_fixed(value: i32) -> f32 {
    value as f32 / PIXEL as f32
}

/// Snap a 26.6 value to the nearest pixel boundary.
///
/// ```
/// use font_shape::grid_fit;
///
/// assert_eq!(grid_fit(100), 128);   // 1.5625 px -> 2 px
/// assert_eq!(grid_fit(40), 64);     // 0.625 px  -> 1 px
/// assert_eq!(grid_fit(30), 0);      // 0.469 px  -> 0 px
/// ```
#[must_use]
pub const fn grid_fit(value: i32) -> i32 {
    let snapped = if value >= 0 {
        (value + (PIXEL / 2)) / PIXEL
    } else {
        -((-value + (PIXEL / 2)) / PIXEL)
    };
    snapped * PIXEL
}

/// The scale from font units to 26.6 fixed-point units: `size_px · 64 / upem`.
///
/// # Errors
/// [`ShapeError::DegenerateUnitsPerEm`] when `units_per_em` is zero, and
/// [`ShapeError::InvalidScale`] when `size_px` is zero, negative, NaN, or
/// infinite — every one of which would produce an empty or infinite glyph.
///
/// ```
/// use font_shape::fixed_scale;
///
/// // A 2048-unit em at 24 px: one font unit is 24/2048 px, or 0.75 of a
/// // 26.6 unit.
/// let s = fixed_scale(24.0, 2048).unwrap();
/// assert!((s - 0.75).abs() < 1e-6, "{s}");
/// ```
#[allow(clippy::result_large_err)]
pub fn fixed_scale(size_px: f32, units_per_em: u16) -> Result<f32, ShapeError> {
    if units_per_em == 0 {
        return Err(ShapeError::DegenerateUnitsPerEm);
    }
    if !size_px.is_finite() || size_px <= 0.0 {
        return Err(ShapeError::InvalidScale { scale: size_px });
    }
    Ok(size_px * PIXEL as f32 / f32::from(units_per_em))
}

/// A glyph's coverage: 8 bits per pixel, with the placement a caller needs to
/// put it back into a text run.
///
/// ```
/// use font_shape::{rasterize_outline, GlyphBitmap};
///
/// let mut outline = font_model::Outline::new();
/// let mut p = font_model::Path::starting_at(0.0, 0.0);
/// p.line_to(1000.0, 0.0);
/// p.line_to(1000.0, 1000.0);
/// p.line_to(0.0, 1000.0);
/// p.close();
/// outline.push_contour(p);
///
/// // A 1000-upem glyph drawn 10 px tall.
/// let bmp = rasterize_outline(&outline, 1000, 10.0, font_shape::Hinting::None).unwrap();
/// assert_eq!(bmp.height(), 10);
/// assert_eq!(bmp.width(), 10);
/// assert_eq!(bmp.left(), 0);
/// assert_eq!(bmp.top(), 10, "the ink starts at the baseline");
/// assert_eq!(bmp.coverage().len(), 100);
/// // It is a solid 10x10 block, so the ink is the full area.
/// assert!((bmp.ink() - 100.0).abs() < 1e-6, "{}", bmp.ink());
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct GlyphBitmap {
    width: u32,
    height: u32,
    left: i32,
    top: i32,
    coverage: Vec<u8>,
}

impl GlyphBitmap {
    /// An empty bitmap: no pixels, no ink, at the origin.
    #[must_use]
    pub const fn empty() -> Self {
        GlyphBitmap {
            width: 0,
            height: 0,
            left: 0,
            top: 0,
            coverage: Vec::new(),
        }
    }

    /// Build from raw parts. The mask must be exactly `width × height` long.
    #[must_use]
    pub const fn new(width: u32, height: u32, left: i32, top: i32, coverage: Vec<u8>) -> Self {
        GlyphBitmap {
            width,
            height,
            left,
            top,
            coverage,
        }
    }

    /// Width in pixels.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Height in pixels.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Columns from the pen position to the bitmap's left edge. May be
    /// negative: a glyph can overhang to the left of its origin.
    #[must_use]
    pub const fn left(&self) -> i32 {
        self.left
    }

    /// Rows from the baseline up to the bitmap's top edge.
    #[must_use]
    pub const fn top(&self) -> i32 {
        self.top
    }

    /// The coverage bytes, row-major from the top-left.
    #[must_use]
    pub fn coverage(&self) -> &[u8] {
        &self.coverage
    }

    /// Consume the bitmap and take its coverage bytes.
    #[must_use]
    pub fn into_coverage(self) -> Vec<u8> {
        self.coverage
    }

    /// True when there are no pixels at all — a blank glyph, a zero-size
    /// raster, or a space.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0 || self.coverage.is_empty()
    }

    /// The total ink, in pixels: `coverage / 255` summed.
    ///
    /// For a shape this is its area, which is the property
    /// [`fill_path`](crate::fill_path) is built to hold.
    #[must_use]
    pub fn ink(&self) -> f64 {
        mask_coverage(&self.coverage) / 255.0
    }

    /// The coverage at `(x, y)`, or `0` outside the bitmap.
    ///
    /// ```
    /// use font_shape::GlyphBitmap;
    ///
    /// let b = GlyphBitmap::new(2, 1, 0, 0, vec![10, 200]);
    /// assert_eq!(b.coverage_at(0, 0), 10);
    /// assert_eq!(b.coverage_at(1, 0), 200);
    /// assert_eq!(b.coverage_at(2, 0), 0, "out of bounds reads as empty");
    /// assert_eq!(b.coverage_at(0, 9), 0);
    /// ```
    #[must_use]
    pub fn coverage_at(&self, x: u32, y: u32) -> u8 {
        if x >= self.width || y >= self.height {
            return 0;
        }
        let idx = y as usize * self.width as usize + x as usize;
        self.coverage.get(idx).copied().unwrap_or(0)
    }

    /// One row of coverage, or an empty slice when the row is off the bitmap.
    #[must_use]
    pub fn row(&self, y: u32) -> &[u8] {
        if y >= self.height {
            return &[];
        }
        let start = y as usize * self.width as usize;
        let end = start + self.width as usize;
        self.coverage.get(start..end).unwrap_or(&[])
    }

    /// The tight ink bounds in bitmap pixels, or `None` when nothing is drawn.
    ///
    /// ```
    /// use font_shape::GlyphBitmap;
    ///
    /// // A 4x3 block whose middle 2x2 is ink.
    /// let b = GlyphBitmap::new(4, 3, 0, 0, vec![
    ///     0, 0, 0, 0,
    ///     0, 255, 255, 0,
    ///     0, 255, 255, 0,
    /// ]);
    /// assert_eq!(b.ink_bounds(), Some((1, 1, 2, 2)));
    /// assert_eq!(GlyphBitmap::empty().ink_bounds(), None);
    /// ```
    #[must_use]
    pub fn ink_bounds(&self) -> Option<(u32, u32, u32, u32)> {
        if self.is_empty() {
            return None;
        }
        let mut x0 = u32::MAX;
        let mut y0 = u32::MAX;
        let mut x1 = 0u32;
        let mut y1 = 0u32;
        let mut any = false;
        for y in 0..self.height {
            for (x, &v) in self.row(y).iter().enumerate() {
                if v > 0 {
                    let x = x as u32;
                    x0 = x0.min(x);
                    x1 = x1.max(x);
                    y0 = y0.min(y);
                    y1 = y1.max(y);
                    any = true;
                }
            }
        }
        if any {
            Some((x0, y0, x1 - x0 + 1, y1 - y0 + 1))
        } else {
            None
        }
    }

    /// Translate the bitmap without touching its pixels.
    #[must_use]
    pub const fn translated(mut self, dx: i32, dy: i32) -> Self {
        self.left += dx;
        self.top += dy;
        self
    }
}

/// The device-space path for a font outline: pixels, y **down**, origin on the
/// baseline at the glyph's x origin.
///
/// The outline's own y-up coordinates are negated, so a glyph that sits above
/// the baseline lands at negative y in the returned path — which is what
/// [`fill_path`](crate::fill_path) wants.
///
/// # Errors
/// [`ShapeError::InvalidScale`] for a non-positive or non-finite `size_px`.
///
/// ```
/// use font_shape::{glyph_path_from_outline, Hinting};
///
/// let mut outline = font_model::Outline::new();
/// let mut p = font_model::Path::starting_at(0.0, 0.0);
/// p.line_to(500.0, 0.0);
/// p.line_to(500.0, 1000.0);
/// p.close();
/// outline.push_contour(p);
///
/// // 1000 units per em at 8 px: the glyph is 4 px wide and 8 px tall, above
/// // the baseline, so its y range is [-8, 0].
/// let path = glyph_path_from_outline(&outline, 1000, 8.0, Hinting::None).unwrap();
/// let b = path.bbox().unwrap();
/// assert_eq!(b.min_x, 0.0);
/// assert_eq!(b.width(), 4.0);
/// assert_eq!(b.min_y, -8.0);
/// assert_eq!(b.max_y, 0.0);
/// ```
#[allow(clippy::result_large_err)]
pub fn glyph_path_from_outline(
    outline: &Outline,
    units_per_em: u16,
    size_px: f32,
    hinting: Hinting,
) -> Result<Path2D, ShapeError> {
    let scale = fixed_scale(size_px, units_per_em)?;
    let mut b = PathBuilder::new();
    for contour in outline.contours() {
        let mut started = false;
        for cmd in contour.commands() {
            let project = |x: f32, y: f32| -> (f32, f32) {
                // `scale` is already 26.6 units per font unit, so
                // the product *is* the fixed-point value: round it there
                // rather than going back through float pixels, which would
                // quantise twice.
                let fx = to_26_6(x * scale);
                let fy = to_26_6(-y * scale);
                let (fx, fy) = if hinting.is_grid_fit() {
                    (grid_fit(fx), grid_fit(fy))
                } else {
                    (fx, fy)
                };
                (from_fixed(fx), from_fixed(fy))
            };
            match *cmd {
                font_model::Command::MoveTo(x, y) => {
                    let (x, y) = project(x, y);
                    b.move_to(x, y);
                    started = true;
                }
                font_model::Command::LineTo(x, y) => {
                    let (x, y) = project(x, y);
                    if started {
                        b.line_to(x, y);
                    }
                }
                font_model::Command::QuadTo(cx, cy, x, y) => {
                    let (cx, cy) = project(cx, cy);
                    let (x, y) = project(x, y);
                    if started {
                        b.quad_to(cx, cy, x, y);
                    }
                }
                font_model::Command::CubicTo(c1x, c1y, c2x, c2y, x, y) => {
                    let (c1x, c1y) = project(c1x, c1y);
                    let (c2x, c2y) = project(c2x, c2y);
                    let (x, y) = project(x, y);
                    if started {
                        b.cubic_to(c1x, c1y, c2x, c2y, x, y);
                    }
                }
                font_model::Command::Close => {
                    if started {
                        b.close();
                    }
                    started = false;
                }
            }
        }
    }
    Ok(b.build())
}

/// The device-space path for a glyph in a font.
///
/// # Errors
/// [`ShapeError::UnknownGlyph`] when the font has no such glyph, plus
/// whatever [`glyph_path_from_outline`] reports.
///
/// ```
/// use font_model::{Font, Glyph, GlyphId, Outline, Path};
/// use font_shape::{glyph_path, Hinting};
///
/// let mut outline = Outline::new();
/// let mut p = Path::starting_at(0.0, 0.0);
/// p.line_to(500.0, 0.0);
/// p.line_to(500.0, 700.0);
/// p.close();
/// outline.push_contour(p);
///
/// let font = Font::builder()
///     .units_per_em(1000)
///     .glyph(Glyph::empty(GlyphId::NOTDEF, 500))
///     .glyph(Glyph::new(GlyphId::new(1), 500, 0, outline))
///     .build()
///     .unwrap();
///
/// // At 12 px the 500-unit glyph is 6 px wide and 8.4 px tall, sitting above
/// // the baseline, so the path's box has a negative top.
/// let path = glyph_path(&font, GlyphId::new(1), 12.0, Hinting::None).unwrap();
/// let b = path.bbox().expect("a glyph with ink has a bbox");
/// assert!(b.max_x - b.min_x <= 6.0 && b.max_x - b.min_x >= 5.9, "{b:?}");
/// assert!(b.max_y <= 0.01 && b.min_y < -8.0, "{b:?}");
/// ```
#[allow(clippy::result_large_err)]
pub fn glyph_path(
    font: &Font,
    glyph: GlyphId,
    size_px: f32,
    hinting: Hinting,
) -> Result<Path2D, ShapeError> {
    let g = font.glyph(glyph).ok_or(ShapeError::UnknownGlyph {
        glyph: glyph.to_u32(),
        glyph_count: font.glyph_count(),
    })?;
    glyph_path_from_outline(g.outline(), font.units_per_em(), size_px, hinting)
}

/// Rasterize an outline into a bitmap placed relative to the baseline.
///
/// The returned bitmap's [`left`](GlyphBitmap::left) and
/// [`top`](GlyphBitmap::top) are relative to the glyph origin: `left` counts
/// columns to the left of the pen, `top` counts rows above the baseline. A
/// glyph with no contours — a space — gives an empty bitmap, not an error.
///
/// # Errors
/// [`ShapeError::InvalidScale`] for a bad size, and
/// [`ShapeError::RasterTooLarge`] when the glyph's own bounding box is larger
/// than [`MAX_MASK_ELEMENTS`] pixels.
#[allow(clippy::result_large_err)]
pub fn rasterize_outline(
    outline: &Outline,
    units_per_em: u16,
    size_px: f32,
    hinting: Hinting,
) -> Result<GlyphBitmap, ShapeError> {
    // Validate the size before touching the outline, so a bad size is the
    // same error whichever entry point the caller used.
    let _ = fixed_scale(size_px, units_per_em)?;
    if outline.contours().is_empty() {
        return Ok(GlyphBitmap::empty());
    }
    let path = glyph_path_from_outline(outline, units_per_em, size_px, hinting)?;
    let Some(bbox) = path.bbox() else {
        return Ok(GlyphBitmap::empty());
    };
    // A non-finite bound means hostile geometry reached the transform. There
    // is no honest raster of it, and the alternative — a width derived from
    // infinity — is a gigabyte allocation. Report no ink instead.
    if bbox.is_empty()
        || !bbox.min_x.is_finite()
        || !bbox.min_y.is_finite()
        || !bbox.max_x.is_finite()
        || !bbox.max_y.is_finite()
    {
        return Ok(GlyphBitmap::empty());
    }
    let left = floor_i32(bbox.min_x);
    let right = ceil_i32(bbox.max_x);
    // Rows above the baseline: the bitmap's top edge.
    let top = ceil_i32(-bbox.min_y);
    // Rows below: its bottom edge, which is at a *negative* offset.
    let bottom = floor_i32(-bbox.max_y);
    let width_i = right - left;
    let height_i = top - bottom;
    if width_i <= 0 || height_i <= 0 {
        return Ok(GlyphBitmap::empty());
    }
    let (Ok(width_u), Ok(height_u)) = (u32::try_from(width_i), u32::try_from(height_i)) else {
        return Ok(GlyphBitmap::empty());
    };
    let pixels = (width_u as usize).saturating_mul(height_u as usize);
    if pixels > MAX_MASK_ELEMENTS {
        return Err(ShapeError::RasterTooLarge {
            pixels,
            limit: MAX_MASK_ELEMENTS,
        });
    }
    // Move the ink so the bitmap's own (0, 0) is its top-left corner. `left`
    // and `top` are pixel counts already, not 26.6 values.
    let placed = path.transformed(&Affine2D::translate(-(left as f32), top as f32));
    let mask = fill_path(&placed, FillRule::NonZero, width_u, height_u)?;
    Ok(GlyphBitmap::new(width_u, height_u, left, top, mask))
}

/// Rasterize a glyph from a font.
///
/// # Errors
/// [`ShapeError::UnknownGlyph`] when the font has no such glyph, plus
/// whatever [`rasterize_outline`] reports.
///
/// ```
/// use font_model::{Font, Glyph, GlyphId, Outline, Path};
/// use font_shape::{rasterize_glyph, Hinting};
///
/// let mut outline = Outline::new();
/// let mut p = Path::starting_at(0.0, 0.0);
/// p.line_to(500.0, 0.0);
/// p.line_to(500.0, 700.0);
/// p.close();
/// outline.push_contour(p);
///
/// let font = Font::builder()
///     .units_per_em(1000)
///     .glyph(Glyph::empty(GlyphId::NOTDEF, 500))
///     .glyph(Glyph::new(GlyphId::new(1), 500, 0, outline))
///     .build()
///     .unwrap();
///
/// let bmp = rasterize_glyph(&font, GlyphId::new(1), 20.0, Hinting::None).unwrap();
/// assert_eq!((bmp.width(), bmp.height()), (10, 14));
/// assert_eq!(bmp.top(), 14);
/// assert!(bmp.ink() > 0.0, "the glyph has ink");
/// assert_eq!(bmp.coverage().len(), (bmp.width() * bmp.height()) as usize);
///
/// // `.notdef` here has no contours, so it has no bitmap at all.
/// let blank = rasterize_glyph(&font, GlyphId::NOTDEF, 16.0, Hinting::None).unwrap();
/// assert!(blank.is_empty());
///
/// // And a glyph past the store is a typed error, not a panic.
/// assert!(rasterize_glyph(&font, GlyphId::new(9), 16.0, Hinting::None).is_err());
/// ```
#[allow(clippy::result_large_err)]
pub fn rasterize_glyph(
    font: &Font,
    glyph: GlyphId,
    size_px: f32,
    hinting: Hinting,
) -> Result<GlyphBitmap, ShapeError> {
    let g = font.glyph(glyph).ok_or(ShapeError::UnknownGlyph {
        glyph: glyph.to_u32(),
        glyph_count: font.glyph_count(),
    })?;
    rasterize_outline(g.outline(), font.units_per_em(), size_px, hinting)
}

/// The device pixel size of the em square at `size_px`.
#[must_use]
pub fn em_size_px(size_px: f32) -> f32 {
    if size_px.is_finite() && size_px > 0.0 {
        size_px
    } else {
        0.0
    }
}

/// Font metrics in device pixels, y down, origin on the baseline.
///
/// Ascender and descender are returned as they are used: `ascent` is a
/// **positive** number of pixels above the baseline, `descent` a positive
/// number below it.
///
/// # Errors
/// [`ShapeError::InvalidScale`] for a non-positive or non-finite size.
///
/// ```
/// use font_shape::scaled_metrics;
///
/// let font = font_model::build_stub_font(&[b'A' as u32]).unwrap();
/// let m = scaled_metrics(&font, 16.0).unwrap();
/// assert!(m.ascent > 0.0 && m.descent > 0.0);
/// // The em square is 16 px tall, so the line fits inside it.
/// assert!(m.line_height >= m.ascent + m.descent);
/// ```
#[allow(clippy::result_large_err)]
pub fn scaled_metrics(font: &Font, size_px: f32) -> Result<DeviceMetrics, ShapeError> {
    if !size_px.is_finite() || size_px <= 0.0 {
        return Err(ShapeError::InvalidScale { scale: size_px });
    }
    let scale = size_px / f32::from(font.units_per_em().max(1));
    Ok(DeviceMetrics::from_font(font.metrics(), scale))
}

/// Font metrics resolved to device pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeviceMetrics {
    /// Pixels above the baseline to the ascender.
    pub ascent: f32,
    /// Pixels below the baseline to the descender, as a positive number.
    pub descent: f32,
    /// Pixels of extra leading between lines.
    pub line_gap: f32,
    /// `ascent + descent + line_gap`: the recommended distance between
    /// consecutive baselines.
    pub line_height: f32,
}

impl DeviceMetrics {
    /// Scale a [`FontMetrics`] by `scale` (pixels per font unit).
    #[must_use]
    pub fn from_font(m: &FontMetrics, scale: f32) -> Self {
        let ascent = f32::from(m.ascender) * scale;
        let descent = -f32::from(m.descender) * scale;
        let line_gap = f32::from(m.line_gap) * scale;
        DeviceMetrics {
            ascent,
            descent,
            line_gap,
            line_height: ascent + descent + line_gap,
        }
    }
}

/// `floor` as an `i32`, saturating.
fn floor_i32(v: f32) -> i32 {
    if v.is_nan() {
        return 0;
    }
    if v <= i32::MIN as f32 {
        return i32::MIN;
    }
    if v >= i32::MAX as f32 {
        return i32::MAX;
    }
    libm::floorf(v) as i32
}

/// `ceil` as an `i32`, saturating.
fn ceil_i32(v: f32) -> i32 {
    if v.is_nan() {
        return 0;
    }
    if v <= i32::MIN as f32 {
        return i32::MIN;
    }
    if v >= i32::MAX as f32 {
        return i32::MAX;
    }
    libm::ceilf(v) as i32
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing,
        clippy::float_cmp
    )]
    use super::{
        em_size_px, fixed_scale, from_fixed, glyph_path, glyph_path_from_outline, grid_fit,
        rasterize_glyph, rasterize_outline, scaled_metrics, to_fixed, DeviceMetrics, GlyphBitmap,
        Hinting, PIXEL,
    };
    use crate::error::ShapeError;
    use alloc::string::ToString;
    use alloc::vec;
    use alloc::vec::Vec;
    use font_model::{Command, Font, Glyph, GlyphId, Outline, Path};

    // Test harness: assertions legitimately panic; the lib target stays
    // lint-clean.

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    /// A square from (x0, y0) to (x1, y1) in font units, y up.
    fn square(x0: f32, y0: f32, x1: f32, y1: f32) -> Outline {
        let mut outline = Outline::new();
        let mut p = Path::starting_at(x0, y0);
        p.line_to(x1, y0);
        p.line_to(x1, y1);
        p.line_to(x0, y1);
        p.close();
        outline.push_contour(p);
        outline
    }

    #[test]
    fn fixed_point_round_trips() {
        assert_eq!(PIXEL, 64);
        for v in [0.0f32, 0.5, 1.0, 1.25, -0.75, 100.0, 1.0 / 64.0] {
            assert!(close(from_fixed(to_fixed(v)), v), "{v}");
        }
        assert_eq!(to_fixed(1.0), PIXEL);
        assert_eq!(from_fixed(PIXEL), 1.0);
        // Every 26.6 value is a multiple of 1/64 px.
        assert_eq!(to_fixed(0.25), 16);
        // Rounding is half-away-from-zero, so the round trip is exact to
        // within half a quantisation step: 1/128 of a pixel.
        assert!(close(from_fixed(to_fixed(-0.007_812_5)), -0.015_625));
    }

    #[test]
    fn fixed_point_saturates_instead_of_wrapping() {
        assert_eq!(to_fixed(f32::NAN), 0);
        assert_eq!(to_fixed(f32::INFINITY), i32::MAX);
        assert_eq!(to_fixed(f32::NEG_INFINITY), i32::MIN);
        assert_eq!(to_fixed(1e18), i32::MAX);
        assert_eq!(to_fixed(-1e18), i32::MIN);
        assert!(from_fixed(i32::MAX).is_finite());
        assert!(from_fixed(i32::MIN).is_finite());
    }

    #[test]
    fn grid_fit_snaps_to_the_pixel_grid() {
        assert_eq!(grid_fit(0), 0);
        assert_eq!(grid_fit(32), 64);
        assert_eq!(grid_fit(-32), -64);
        assert_eq!(grid_fit(100), 128);
        assert_eq!(grid_fit(31), 0);
        assert_eq!(grid_fit(-31), 0);
        assert_eq!(grid_fit(95), 64);
        // Snapping is idempotent.
        for v in [-500, -64, -1, 0, 1, 33, 200] {
            assert_eq!(grid_fit(grid_fit(v)), grid_fit(v), "{v}");
        }
    }

    #[test]
    fn fixed_scale_is_size_times_sixty_four_over_upem() {
        assert!(close(fixed_scale(16.0, 1000).expect("scale"), 1.024));
        assert!(close(fixed_scale(24.0, 2048).expect("scale"), 0.75));
        // Half size, half scale.
        let a = fixed_scale(12.0, 1000).expect("scale");
        let b = fixed_scale(24.0, 1000).expect("scale");
        assert!(close(a, b * 0.5));
    }

    #[test]
    fn fixed_scale_rejects_degenerate_inputs() {
        assert_eq!(
            fixed_scale(16.0, 0).unwrap_err(),
            ShapeError::DegenerateUnitsPerEm
        );
        for bad in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(
                matches!(fixed_scale(bad, 1000), Err(ShapeError::InvalidScale { .. })),
                "{bad}"
            );
        }
    }

    #[test]
    fn glyph_path_flips_y_and_scales() {
        // 1000 upem at 100 px: a 1000-unit square is 100 px, above the
        // baseline, so y runs from -100 to 0.
        let outline = square(0.0, 0.0, 1000.0, 1000.0);
        let path = glyph_path_from_outline(&outline, 1000, 100.0, Hinting::None).expect("path");
        let b = path.bbox().expect("bbox");
        assert_eq!(b.min_x, 0.0);
        assert_eq!(b.min_y, -100.0);
        assert_eq!(b.max_x, 100.0);
        assert_eq!(b.max_y, 0.0);
    }

    #[test]
    fn glyph_path_carries_curves_through() {
        let mut outline = Outline::new();
        let mut p = Path::starting_at(0.0, 0.0);
        p.cubic_to(500.0, 500.0, 1000.0, 500.0, 1000.0, 0.0);
        p.close();
        outline.push_contour(p);
        let path = glyph_path_from_outline(&outline, 1000, 100.0, Hinting::None).expect("path");
        // A quad and a cubic both survive as curves, not polylines.
        assert!(path
            .commands()
            .iter()
            .any(|c| matches!(c, crate::PathCommand::CubicTo(..))));
        let b = path.bbox().expect("bbox");
        assert!(b.min_y < -10.0, "the curve bows up: {b:?}");
    }

    #[test]
    fn grid_fit_snaps_the_whole_outline_to_pixels() {
        let outline = square(0.0, 0.0, 1000.0, 1000.0);
        let path = glyph_path_from_outline(&outline, 1000, 7.3, Hinting::GridFit).expect("path");
        for cmd in path.commands() {
            if let Some((x, y)) = cmd.end_point() {
                assert!(close(x, libm::roundf(x)), "x {x} is not a whole pixel");
                assert!(close(y, libm::roundf(y)), "y {y} is not a whole pixel");
            }
        }
        // And the unhinted version generally is not.
        let raw = glyph_path_from_outline(&outline, 1000, 7.3, Hinting::None).expect("path");
        let snapped = path.bbox().expect("bbox");
        let plain = raw.bbox().expect("bbox");
        // Unhinted, the glyph is 7.3 px — up to the 26.6 quantisation of
        // 1/64 px, which is the fixed-point grid the glyph is snapped to.
        assert!((plain.width() - 7.3).abs() < 0.02, "{plain:?}");
        assert!((plain.height() - 7.3).abs() < 0.02, "{plain:?}");
        assert_eq!(snapped.width(), 7.0, "grid fit rounds 7.3 down to 7");
        assert_eq!(snapped.height(), 7.0, "{snapped:?}");
    }

    #[test]
    fn grid_fit_can_change_the_measured_size_by_a_pixel() {
        // 7.3 px snaps to 7; 7.8 px snaps to 8.
        let outline = square(0.0, 0.0, 1000.0, 1000.0);
        let small = glyph_path_from_outline(&outline, 1000, 7.8, Hinting::GridFit)
            .expect("path")
            .bbox()
            .expect("bbox");
        assert!(close(small.width(), 8.0), "{small:?}");
    }

    #[test]
    fn rasterized_square_is_exact_and_placed() {
        // A 20 px square sitting on the baseline at x = 0.
        let outline = square(0.0, 0.0, 1000.0, 1000.0);
        let bmp = rasterize_outline(&outline, 1000, 20.0, Hinting::None).expect("raster");
        assert_eq!(bmp.width(), 20);
        assert_eq!(bmp.height(), 20);
        assert_eq!(bmp.left(), 0);
        assert_eq!(bmp.top(), 20, "20 rows above the baseline");
        assert!((bmp.ink() - 400.0).abs() < 1e-6, "ink {}", bmp.ink());
        // The baseline row is the bottom row and is solid; above it, nothing.
        assert_eq!(bmp.coverage_at(0, 19), 255);
        assert_eq!(bmp.coverage_at(19, 0), 255);
        assert!(!bmp.is_empty());
    }

    #[test]
    fn bitmap_placement_follows_the_glyph_origin() {
        // A square from x = 100..300 of a 1000-unit glyph at 10 px: 10 px
        // wide, starting 1 px right of the pen.
        let outline = square(100.0, 0.0, 300.0, 1000.0);
        let bmp = rasterize_outline(&outline, 1000, 10.0, Hinting::None).expect("raster");
        assert_eq!(bmp.left(), 1);
        assert_eq!(bmp.width(), 2);
        assert_eq!(bmp.top(), 10);
        // A negative overhang: ink left of the origin.
        let outline = square(-250.0, 0.0, 250.0, 1000.0);
        let bmp = rasterize_outline(&outline, 1000, 10.0, Hinting::None).expect("raster");
        // 2.5 px of overhang left and right, so the bitmap spans six columns
        // even though the ink covers five of them.
        assert_eq!(bmp.left(), -3);
        assert_eq!(bmp.width(), 6);
        assert_eq!(bmp.coverage_at(0, 0), 128, "half-covered left edge");
        assert_eq!(bmp.coverage_at(5, 0), 128, "half-covered right edge");
        assert!((bmp.ink() - 50.0).abs() < 0.05, "ink {}", bmp.ink());
    }

    #[test]
    fn below_baseline_ink_extends_the_bitmap_downward() {
        // From -300 to +100 units at 10 px the ink spans y = -3..1 px, so it
        // reaches 1 px above the baseline and 3 px below it: `top` counts the
        // rows above, `height` the whole box.
        let outline = square(0.0, -300.0, 1000.0, 100.0);
        let bmp = rasterize_outline(&outline, 1000, 10.0, Hinting::None).expect("raster");
        assert_eq!(bmp.top(), 1);
        assert_eq!(bmp.height(), 4);
        assert_eq!(bmp.coverage_at(0, 0), 255, "above the baseline");
        assert_eq!(bmp.coverage_at(0, 3), 255, "below the baseline");
        assert!((bmp.ink() - 40.0).abs() < 0.05, "ink {}", bmp.ink());
    }

    #[test]
    fn empty_outline_is_an_empty_bitmap_not_an_error() {
        let bmp = rasterize_outline(&Outline::new(), 1000, 12.0, Hinting::None).expect("raster");
        assert!(bmp.is_empty());
        assert_eq!(bmp.width(), 0);
        assert_eq!(bmp.ink(), 0.0);
        // A contour that is a single point has no area either.
        // A contour that is one point and nothing else. It has a bounding box
        // but no area, so the honest answer is a bitmap with pixels and no ink.
        let mut outline = Outline::new();
        let mut p = Path::starting_at(100.0, 100.0);
        p.close();
        outline.push_contour(p);
        let bmp = rasterize_outline(&outline, 1000, 12.0, Hinting::None).expect("raster");
        assert!(bmp.ink() == 0.0, "a lone point has no ink");
        assert!(bmp.ink_bounds().is_none(), "and no ink bounds");
    }

    #[test]
    fn raster_size_errors_are_typed() {
        let outline = square(0.0, 0.0, 1000.0, 1000.0);
        assert!(matches!(
            rasterize_outline(&outline, 1000, 0.0, Hinting::None),
            Err(ShapeError::InvalidScale { .. })
        ));
        assert!(matches!(
            rasterize_outline(&outline, 0, 12.0, Hinting::None),
            Err(ShapeError::DegenerateUnitsPerEm)
        ));
        // A glyph 100 000 px tall is refused rather than allocated.
        assert!(matches!(
            rasterize_outline(&outline, 1000, 100_000.0, Hinting::None),
            Err(ShapeError::RasterTooLarge { .. })
        ));
    }

    #[test]
    fn glyph_ink_tracks_the_outline_area() {
        // A 500 x 400 unit glyph at 20 px is a 10 x 8 px rectangle: 80 px of
        // ink. The analytic filler makes that exact.
        let outline = square(0.0, 0.0, 500.0, 400.0);
        let bmp = rasterize_outline(&outline, 1000, 20.0, Hinting::None).expect("raster");
        assert_eq!(bmp.width(), 10);
        assert_eq!(bmp.height(), 8);
        assert!((bmp.ink() - 80.0).abs() < 1e-6, "ink {}", bmp.ink());
        // Scale by 3: 9x the area.
        let big = rasterize_outline(&outline, 1000, 60.0, Hinting::None).expect("raster");
        assert!((big.ink() - 80.0 * 9.0).abs() < 1e-6, "ink {}", big.ink());
    }

    #[test]
    fn rasterize_glyph_goes_through_the_font() {
        let mut outline = square(0.0, 0.0, 1000.0, 1000.0);
        let font = Font::builder()
            .units_per_em(1000)
            .glyph(Glyph::empty(GlyphId::NOTDEF, 500))
            .glyph(Glyph::new(GlyphId::new(1), 1000, 0, outline.clone()))
            .build()
            .expect("font");
        let bmp = rasterize_glyph(&font, GlyphId::new(1), 10.0, Hinting::None).expect("raster");
        assert_eq!(bmp.width(), 10);
        assert!((bmp.ink() - 100.0).abs() < 1e-6, "ink {}", bmp.ink());
        outline.contours_mut().clear();
        assert_eq!(outline.contours().len(), 0);
        // A missing glyph is a typed error, with the count in it.
        match rasterize_glyph(&font, GlyphId::new(9), 10.0, Hinting::None).unwrap_err() {
            ShapeError::UnknownGlyph { glyph, glyph_count } => {
                assert_eq!(glyph, 9);
                assert_eq!(glyph_count, 2);
            }
            other => panic!("expected UnknownGlyph, got {other:?}"),
        }
        assert_eq!(
            ShapeError::UnknownGlyph {
                glyph: 9,
                glyph_count: 2
            }
            .to_string(),
            "glyph 9 requested but the font holds only 2"
        );
    }

    #[test]
    fn glyph_path_from_a_font() {
        let font = Font::builder()
            .units_per_em(1000)
            .glyph(Glyph::empty(GlyphId::NOTDEF, 500))
            .glyph(Glyph::new(
                GlyphId::new(1),
                1000,
                0,
                square(0.0, 0.0, 500.0, 500.0),
            ))
            .build()
            .expect("font");
        let path = glyph_path(&font, GlyphId::new(1), 100.0, Hinting::None).expect("path");
        let b = path.bbox().expect("bbox");
        assert_eq!(b.width(), 50.0);
        assert_eq!(b.height(), 50.0);
        assert_eq!(b.min_y, -50.0, "the ink sits above the baseline");
        // `.notdef` is blank in this font, so it has no path to speak of.
        let blank = glyph_path(&font, GlyphId::NOTDEF, 12.0, Hinting::None).expect("path");
        assert!(blank.bbox().is_none() || blank.bbox().is_some_and(|r| r.is_empty()));
        // A glyph past the store is an error.
        assert!(glyph_path(&font, GlyphId::new(7), 12.0, Hinting::None).is_err());
    }

    #[test]
    fn bitmap_accessors_are_bounds_checked() {
        let bmp = GlyphBitmap::new(
            4,
            3,
            1,
            2,
            vec![
                0, 0, 0, 0, //
                0, 255, 255, 0, //
                0, 255, 255, 0, //
            ],
        );
        assert_eq!(bmp.coverage().len(), 12);
        assert_eq!(bmp.row(0).len(), 4);
        assert_eq!(bmp.row(1)[1], 255);
        assert!(bmp.row(3).is_empty());
        assert_eq!(bmp.coverage_at(3, 2), 0);
        assert_eq!(bmp.ink_bounds(), Some((1, 1, 2, 2)));
        assert!((bmp.ink() - 4.0).abs() < 1e-9);
        let moved = bmp.clone().translated(5, -5);
        assert_eq!(moved.left(), 6);
        assert_eq!(moved.top(), -3);
        assert_eq!(
            moved.coverage(),
            bmp.coverage(),
            "translation moves nothing else"
        );
        assert_eq!(moved.into_coverage().len(), 12);
        assert_eq!(GlyphBitmap::default(), GlyphBitmap::empty());
        // A mask shorter than its declared size must not panic.
        let short = GlyphBitmap::new(100, 100, 0, 0, Vec::new());
        assert_eq!(short.coverage_at(50, 50), 0);
        assert!(short.row(50).is_empty());
        assert_eq!(short.ink_bounds(), None);
        assert!(short.ink() == 0.0);
    }

    #[test]
    fn ink_bounds_is_none_for_a_blank_glyph() {
        let blank = GlyphBitmap::new(4, 4, 0, 0, vec![0u8; 16]);
        assert_eq!(blank.ink_bounds(), None);
        assert!(!blank.is_empty(), "it has pixels, just no ink");
        // A single fully covered pixel is a 1x1 bound.
        let dot = GlyphBitmap::new(2, 2, 0, 0, vec![0, 0, 0, 255]);
        assert_eq!(dot.ink_bounds(), Some((1, 1, 1, 1)));
    }

    #[test]
    fn hinting_modes_toggle_and_report() {
        assert_eq!(Hinting::default(), Hinting::None);
        assert!(!Hinting::None.is_grid_fit());
        assert!(Hinting::GridFit.is_grid_fit());
        assert_eq!(Hinting::None.toggled(), Hinting::GridFit);
        assert_eq!(Hinting::GridFit.toggled(), Hinting::None);
    }

    #[test]
    fn scaled_metrics_are_in_pixels() {
        let m = font_model::FontMetrics {
            ascender: 800,
            descender: -200,
            line_gap: 100,
            ..font_model::FontMetrics::default()
        };
        let d = DeviceMetrics::from_font(&m, 0.02);
        assert!(close(d.ascent, 16.0));
        assert!(close(d.descent, 4.0));
        assert!(close(d.line_gap, 2.0));
        assert!(close(d.line_height, 22.0));
        // Through the font.
        let font = font_model::build_stub_font(&[b'A' as u32]).expect("font");
        let d = scaled_metrics(&font, 1000.0).expect("metrics");
        assert!(close(d.ascent, f32::from(font.metrics().ascender)));
        assert!(matches!(
            scaled_metrics(&font, 0.0),
            Err(ShapeError::InvalidScale { .. })
        ));
        assert!(matches!(
            scaled_metrics(&font, f32::NAN),
            Err(ShapeError::InvalidScale { .. })
        ));
    }

    #[test]
    fn em_size_is_the_size_itself_when_sane() {
        assert_eq!(em_size_px(12.0), 12.0);
        assert_eq!(em_size_px(0.0), 0.0);
        assert_eq!(em_size_px(-1.0), 0.0);
        assert_eq!(em_size_px(f32::NAN), 0.0);
        assert_eq!(em_size_px(f32::INFINITY), 0.0);
    }

    #[test]
    fn non_finite_outline_coordinates_do_not_panic() {
        // A NaN coordinate never reaches the path: `to_26_6` maps NaN to zero,
        // so the degenerate segment is simply absent.
        let mut outline = Outline::new();
        let mut p = Path::starting_at(0.0, 0.0);
        p.line_to(f32::NAN, 100.0);
        p.line_to(500.0, 500.0);
        p.close();
        outline.push_contour(p);
        let bmp = rasterize_outline(&outline, 1000, 10.0, Hinting::None).expect("raster");
        assert!(bmp.ink() > 0.0, "the good part still draws");

        // An infinite coordinate saturates the fixed-point conversion to
        // `i32::MAX`, so the glyph's box becomes enormous — and an enormous
        // glyph is a typed refusal, not an allocation. Either outcome is
        // total; neither panics.
        let mut wide = Outline::new();
        let mut q = Path::starting_at(0.0, 0.0);
        q.line_to(f32::INFINITY, 100.0);
        q.close();
        wide.push_contour(q);
        match rasterize_outline(&wide, 1000, 10.0, Hinting::None) {
            Ok(bmp) => assert!(bmp.is_empty() || bmp.ink() >= 0.0),
            Err(e) => assert!(
                matches!(e, ShapeError::RasterTooLarge { .. }),
                "unexpected error: {e:?}"
            ),
        }
    }

    #[test]
    fn contours_without_a_move_to_are_skipped() {
        // A contour whose first command is a line has no start point to
        // transform from, so it contributes nothing rather than guessing.
        let mut outline = Outline::new();
        let mut p = Path::new();
        p.push(Command::LineTo(100.0, 100.0));
        p.push(Command::Close);
        outline.push_contour(p);
        let bmp = rasterize_outline(&outline, 1000, 10.0, Hinting::None).expect("raster");
        assert!(bmp.ink() == 0.0, "ink {}", bmp.ink());
    }

    #[test]
    fn a_glyph_scales_linearly_with_size() {
        let outline = square(0.0, 0.0, 1000.0, 1000.0);
        let mut prev: Option<f64> = None;
        for size in [1.0f32, 2.0, 4.0, 8.0, 16.0, 32.0] {
            let bmp = rasterize_outline(&outline, 1000, size, Hinting::None).expect("raster");
            assert_eq!(bmp.width() as f32, size, "width at {size}");
            assert_eq!(bmp.height() as f32, size, "height at {size}");
            assert!(close(bmp.top() as f32, size), "top {}", bmp.top());
            if let Some(p) = prev {
                assert!((bmp.ink() / p - 4.0).abs() < 1e-9, "doubling the size");
            }
            prev = Some(bmp.ink());
        }
    }
}
