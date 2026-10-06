//! The scanline rasteriser: exact analytic coverage.
//!
//! `fill_path` walks each pixel row, finds where the path's edges cross it,
//! determines the inside spans from the fill rule, and computes the **exact**
//! area each span covers in each pixel column. The result is an 8-bit coverage
//! mask, one byte per pixel, `0..=255`.
//!
//! # Why exact, and what that buys
//!
//! The alternative — supersampling a point-in-path test — is approximate, and
//! its error is worst exactly where a glyph rasteriser is most sensitive: thin
//! stems and small sizes. Analytic area gives two properties that are
//! directly testable:
//!
//! - An **axis-aligned rectangle** gets exactly the coverage its geometry
//!   implies: a rectangle on integer boundaries is all `255`, the same
//!   rectangle shifted half a pixel is exactly half covered on each edge.
//! - The **sum of a mask's coverage** equals the polygon's area to within the
//!   curve-flattening tolerance, so `mask_sum / 255 ≈ path_area` for any shape.
//!   That is the single number that bounds every antialiasing claim this crate
//!   makes.
//!
//! Curves are flattened first, so the exactness is with respect to the
//! *polygon*; the flattening tolerance sets how far a mask can be from the true
//! curve, and it is chosen from the geometry rather than hard-coded.
//!
//! # Coordinates
//!
//! Device space with **y down**: pixel `(x, y)` covers the unit square
//! `[x, x+1) × [y, y+1)`. A glyph is flipped into that space before
//! rasterisation, which is why a font's y-up outlines and a bitmap's y-down
//! rows do not need to be reconciled anywhere else.

use alloc::vec;
use alloc::vec::Vec;

use crate::error::ShapeError;
use crate::path::Path2D;

/// Which rule decides whether a point is inside the path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FillRule {
    /// Inside where the winding number is non-zero. The default, and what a
    /// glyph wants: a glyph's outer and inner contours usually run the same
    /// way, so non-zero makes the counter the same colour.
    #[default]
    NonZero,
    /// Inside where the crossing count is odd. This is the rule that punches a
    /// hole in a nested contour regardless of its direction.
    EvenOdd,
}

/// The largest mask this crate will allocate: 4096 × 4096 = 16 MiB of coverage.
pub const MAX_MASK_ELEMENTS: usize = 16_777_216;

/// The flattener's tolerance for fill, as a fraction of a pixel.
///
/// A quarter of a pixel would be visible; 1/64th keeps a curve's flattened
/// area within a few hundredths of a percent of the true area, well inside the
/// 1 % the tests allow.
const FILL_TOLERANCE: f32 = 1.0 / 64.0;

/// Fill `path` into an 8-bit coverage mask `width` × `height`.
///
/// `data.len()` is exactly `width × height`, row-major from the top-left.
///
/// Returns `Ok(Vec::new())` for a zero-sized raster, and an empty mask for a
/// path that covers nothing.
///
/// # Errors
/// [`ShapeError::RasterTooLarge`] when `width × height` exceeds
/// [`MAX_MASK_ELEMENTS`] — a guard against a size argument that would allocate
/// gigabytes.
///
/// ```
/// use font_shape::{fill_path, FillRule, PathBuilder};
///
/// let mut b = PathBuilder::new();
/// b.rect(4.0, 4.0, 8.0, 8.0);          // exactly 8x8 pixels
/// let path = b.build();
/// let mask = fill_path(&path, FillRule::NonZero, 16, 16).unwrap();
///
/// assert_eq!(mask.len(), 16 * 16);
/// // Every pixel of the 8x8 block is fully covered...
/// for y in 4..12 {
///     for x in 4..12 {
///         assert_eq!(mask[y * 16 + x], 255, "({x}, {y})");
///     }
/// }
/// // ...and nothing outside it is.
/// assert_eq!(mask[3 * 16 + 4], 0);
/// assert_eq!(mask[0], 0);
/// ```
#[allow(clippy::result_large_err)]
pub fn fill_path(
    path: &Path2D,
    fill_rule: FillRule,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, ShapeError> {
    let w = usize::try_from(width).unwrap_or(usize::MAX);
    let h = usize::try_from(height).unwrap_or(usize::MAX);
    if w == 0 || h == 0 {
        return Ok(Vec::new());
    }
    let pixels = w.saturating_mul(h);
    if pixels > MAX_MASK_ELEMENTS {
        return Err(ShapeError::RasterTooLarge {
            pixels,
            limit: MAX_MASK_ELEMENTS,
        });
    }
    let mut mask = vec![0u8; pixels];
    let edges = collect_edges(path);
    if edges.is_empty() {
        return Ok(mask);
    }
    let coverage = rasterize(&edges, fill_rule, w, h);
    for (slot, &c) in mask.iter_mut().zip(coverage.iter()) {
        *slot = c;
    }
    Ok(mask)
}

/// The total coverage of a mask, in 0..255 units. Dividing by 255 gives the
/// ink area in pixels.
#[must_use]
pub fn mask_coverage(mask: &[u8]) -> f64 {
    mask.iter().map(|&v| f64::from(v)).sum::<f64>()
}

/// A flattened line segment, as the rasteriser sees it.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Edge {
    /// x at `y_min`.
    x0: f32,
    y0: f32,
    /// x at `y_max`.
    x1: f32,
    y1: f32,
    /// The winding contribution: +1 when the original edge went down in y.
    dir: i32,
}

impl Edge {
    /// The lower y of the edge's active span.
    const fn y_min(&self) -> f32 {
        self.y0
    }

    /// The upper y of the edge's active span.
    const fn y_max(&self) -> f32 {
        self.y1
    }

    /// x at height `y`, which must lie within the edge's span.
    fn x_at(&self, y: f32) -> f32 {
        let dy = self.y1 - self.y0;
        if dy == 0.0 {
            self.x0
        } else {
            self.x0 + (self.x1 - self.x0) * (y - self.y0) / dy
        }
    }
}

/// Flatten a path into edges, dropping horizontal and non-finite segments.
fn collect_edges(path: &Path2D) -> Vec<Edge> {
    let polys = path.flatten(FILL_TOLERANCE);
    let mut out = Vec::new();
    for poly in polys {
        for pair in poly.windows(2) {
            let (Some(a), Some(b)) = (pair.first(), pair.get(1)) else {
                continue;
            };
            if !a.0.is_finite() || !a.1.is_finite() || !b.0.is_finite() || !b.1.is_finite() {
                continue;
            }
            // A horizontal edge bounds no area, and the half-open row rule
            // handles the rest.
            if (a.1 - b.1).abs() < 1e-12 {
                continue;
            }
            let dir = if b.1 > a.1 { 1 } else { -1 };
            let (lo, hi) = if dir > 0 { (*a, *b) } else { (*b, *a) };
            out.push(Edge {
                x0: lo.0,
                y0: lo.1,
                x1: hi.0,
                y1: hi.1,
                dir,
            });
        }
    }
    out
}

/// The per-pixel coverage of a set of edges.
fn rasterize(edges: &[Edge], rule: FillRule, width: usize, height: usize) -> Vec<u8> {
    let mut out = vec![0u8; width.saturating_mul(height)];
    // Bucket edges by the first pixel row they touch, so each row only walks
    // its own edges.
    let mut rows: Vec<Vec<usize>> = vec![Vec::new(); height];
    for (i, e) in edges.iter().enumerate() {
        let first = floor_to_i32(e.y_min());
        let last = ceil_to_i32(e.y_max());
        let mut r = first.max(0);
        while r < last.min(height as i32) {
            if let Some(bucket) = rows.get_mut(r as usize) {
                bucket.push(i);
            }
            r += 1;
        }
    }
    for (y, bucket) in rows.iter().enumerate() {
        if bucket.is_empty() {
            continue;
        }
        let row_edges: Vec<Edge> = bucket
            .iter()
            .filter_map(|i| edges.get(*i).copied())
            .collect();
        rasterize_row(&mut out, &row_edges, rule, y, width);
    }
    out
}

/// Fill one pixel row, splitting it into bands where the edge set is stable.
///
/// The naive approach — order the row's edges once and integrate each span
/// from the row's top to its bottom — is wrong for a curve. A circle's top arc
/// crosses a row several times, so the edge that is "left" at the row's top is
/// not the edge that is left at its bottom, and clamping a span to one edge's
/// y range throws away most of the row's ink.
///
/// Splitting the row at every active edge's endpoints makes each band's edge
/// set fixed, and within a band the x-ordering cannot change (two straight
/// edges that swap order would have to cross, and a crossing inside a band
/// would have been a split point). Ordering by the band's midline is then
/// exact for that band.
fn rasterize_row(out: &mut [u8], row_edges: &[Edge], rule: FillRule, y: usize, width: usize) {
    let row_top = y as f32;
    let row_bottom = row_top + 1.0;

    // Band breakpoints: the row's edges, plus every place one of them starts
    // or stops.
    let mut breaks: Vec<f32> = vec![row_top, row_bottom];
    for e in row_edges {
        for v in [e.y_min(), e.y_max()] {
            if v > row_top && v < row_bottom {
                breaks.push(v);
            }
        }
    }
    breaks.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    breaks.dedup_by(|a, b| (*a - *b).abs() < f32::EPSILON);

    for w in breaks.windows(2) {
        let (Some(&t0), Some(&t1)) = (w.first(), w.get(1)) else {
            continue;
        };
        if t1 <= t0 {
            continue;
        }
        let midline = (t0 + t1) * 0.5;
        let active: Vec<Edge> = row_edges
            .iter()
            .copied()
            .filter(|e| e.y_min() <= midline && midline <= e.y_max())
            .collect();
        if active.is_empty() {
            continue;
        }
        for (left, right) in inside_spans(&active, rule, midline) {
            add_band(out, &(left, right), t0, t1, y, width);
        }
    }
}

/// The inside spans of one pixel row: `(left edge, right edge)` pairs.
///
/// `midline` is the y at which the edges are ordered. Ordering by any value
/// that is constant across the row's edges — their own midpoints, say — gets
/// curved rows wrong: two edges of a circle that meet at the row's top can be
/// the wrong way round by the bottom. The midline is inside the row, so its
/// ordering is the ordering that holds.
fn inside_spans(row_edges: &[Edge], rule: FillRule, midline: f32) -> Vec<(Edge, Edge)> {
    let mut active: Vec<Edge> = row_edges.to_vec();
    active.sort_by(|a, b| {
        a.x_at(midline)
            .partial_cmp(&b.x_at(midline))
            .unwrap_or(core::cmp::Ordering::Equal)
            .then(a.dir.cmp(&b.dir))
    });
    let mut out = Vec::new();
    let mut winding = 0i32;
    let mut parity = 0u32;
    // The left edge of the span currently being built.
    let mut open: Option<Edge> = None;
    for e in active {
        let was_inside = match rule {
            FillRule::NonZero => winding != 0,
            FillRule::EvenOdd => parity % 2 == 1,
        };
        match rule {
            FillRule::NonZero => winding += e.dir,
            FillRule::EvenOdd => parity = parity.wrapping_add(1),
        }
        let is_inside = match rule {
            FillRule::NonZero => winding != 0,
            FillRule::EvenOdd => parity % 2 == 1,
        };
        match (was_inside, is_inside) {
            (false, true) => open = Some(e),
            (true, false) => {
                if let Some(left) = open.take() {
                    out.push((left, e));
                }
            }
            _ => {}
        }
    }
    out
}

/// Add one span's exact area to the mask, over the y band `[ya, yb)`.
fn add_band(out: &mut [u8], span: &(Edge, Edge), ya: f32, yb: f32, y: usize, width: usize) {
    let (left, right) = span;
    if yb <= ya {
        return;
    }
    // The span's x extent at the ends of the band bounds the columns it can
    // touch. Using the midpoint x would clip a slanted band short, so the
    // bounds are the min and max over the band.
    let lx0 = left.x_at(ya);
    let lx1 = left.x_at(yb);
    let rx0 = right.x_at(ya);
    let rx1 = right.x_at(yb);
    let min_x = lx0.min(lx1).min(rx0.min(rx1));
    let max_x = lx0.max(lx1).max(rx0.max(rx1));
    let first = floor_to_i32(min_x);
    let last = ceil_to_i32(max_x);
    let mut cx = first.max(0);
    while cx < last {
        let col = cx as f32;
        let area = span_column_area(left, right, ya, yb, col);
        if area > 0.0 {
            // `libm` rather than `f32::round`: the crate is `no_std`, and
            // `round`/`clamp` are `std` methods on floats.
            let cov = libm::roundf(area * 255.0).max(0.0).min(255.0) as u32;
            let v = cov.min(255) as u8;
            let idx = y.saturating_mul(width).saturating_add(cx as usize);
            if let Some(slot) = out.get_mut(idx) {
                // A row is filled band by band, so bands accumulate rather
                // than overwrite. Saturating, because the bands are disjoint
                // in y and cannot exceed full coverage.
                *slot = slot.saturating_add(v);
            }
        }
        cx += 1;
    }
}

/// The exact area a span covers in column `[cx, cx+1)` over `y ∈ [ya, yb)`.
///
/// The integrand is
/// `clamp(min(lx(y), rx(y), cx+1) − max(lx(y), rx(y), cx), 0, 1)`, which is
/// piecewise linear in `y` with breakpoints wherever an edge crosses `cx` or
/// `cx+1`. Splitting at those breakpoints and summing trapezoids is exact.
fn span_column_area(left: &Edge, right: &Edge, ya: f32, yb: f32, cx: f32) -> f32 {
    let mut breaks = vec![ya, yb];
    for e in [left, right] {
        for edge_x in [cx, cx + 1.0] {
            if let Some(t) = edge_crosses_at(e, edge_x, ya, yb) {
                breaks.push(t);
            }
        }
    }
    breaks.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let mut total = 0.0f32;
    for w in breaks.windows(2) {
        let (Some(t0), Some(t1)) = (w.first(), w.get(1)) else {
            continue;
        };
        if *t1 <= *t0 {
            continue;
        }
        let f0 = span_fraction(left, right, *t0, cx);
        let f1 = span_fraction(left, right, *t1, cx);
        total += (f0 + f1) * 0.5 * (t1 - t0);
    }
    total
}

/// The height at which `edge` crosses `x`, if it does within `[ya, yb)`.
fn edge_crosses_at(edge: &Edge, x: f32, ya: f32, yb: f32) -> Option<f32> {
    let a = edge.x_at(ya);
    let b = edge.x_at(yb);
    if (a - x) * (b - x) > 0.0 || a == b {
        return None;
    }
    let dx = b - a;
    if dx == 0.0 {
        return None;
    }
    let t = ya + (x - a) / dx * (yb - ya);
    if t > ya && t < yb {
        Some(t)
    } else {
        None
    }
}

/// The span's horizontal overlap with `[cx, cx+1)` at height `y`, in `0..=1`.
fn span_fraction(left: &Edge, right: &Edge, y: f32, cx: f32) -> f32 {
    let a = left.x_at(y);
    let b = right.x_at(y);
    // A span is between its two edges; if the polygon self-crosses the two
    // can swap order mid-row, so sort before taking the ends.
    let (lo_edge, hi_edge) = if a <= b { (a, b) } else { (b, a) };
    let lo = lo_edge.max(cx);
    let hi = hi_edge.min(cx + 1.0);
    // `.max`/`.min` rather than `clamp`: `clamp` panics on a NaN bound, and a
    // NaN coordinate can reach here.
    #[allow(clippy::manual_clamp)] // `clamp` panics on a NaN bound; a NaN can reach here.
    let frac = (hi - lo).max(0.0).min(1.0);
    frac
}

/// `floor` as an `i32`, saturating at the type's bounds.
fn floor_to_i32(v: f32) -> i32 {
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

/// `ceil` as an `i32`, saturating at the type's bounds.
fn ceil_to_i32(v: f32) -> i32 {
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

/// The tight pixel bounds a path occupies in a raster of the given size, or
/// `None` when it covers nothing.
///
/// Used by [`GlyphBitmap`] placement](crate::GlyphBitmap) and by the tests
/// that check a mask against its geometry.
#[must_use]
pub fn path_pixel_bounds(path: &Path2D, width: u32, height: u32) -> Option<(i32, i32, i32, i32)> {
    let bbox = path.bbox()?;
    if bbox.is_empty() {
        return None;
    }
    let x0 = floor_to_i32(bbox.min_x).max(0);
    let y0 = floor_to_i32(bbox.min_y).max(0);
    let x1 = ceil_to_i32(bbox.max_x).min(width as i32);
    let y1 = ceil_to_i32(bbox.max_y).min(height as i32);
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    Some((x0, y0, x1 - x0, y1 - y0))
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
        collect_edges, fill_path, mask_coverage, path_pixel_bounds, Edge, FillRule,
        MAX_MASK_ELEMENTS,
    };
    use crate::path::PathBuilder;
    use crate::ShapeError;

    // Test harness: assertions legitimately panic; the lib target stays
    // lint-clean.

    /// The total ink of a mask, in pixels.
    fn ink(mask: &[u8]) -> f64 {
        mask_coverage(mask) / 255.0
    }

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    fn rect(x: f32, y: f32, w: f32, h: f32) -> crate::Path2D {
        let mut b = PathBuilder::new();
        b.rect(x, y, w, h);
        b.build()
    }

    fn donut(outer: f32, inner: f32) -> crate::Path2D {
        // Two concentric circles wound the *same* way: the case that
        // distinguishes the rules.
        let mut b = PathBuilder::new();
        b.circle(50.0, 50.0, outer);
        b.circle(50.0, 50.0, inner);
        b.build()
    }

    #[test]
    fn axis_aligned_rectangle_at_integer_offsets_is_exact() {
        let mask = fill_path(&rect(4.0, 4.0, 8.0, 8.0), FillRule::NonZero, 16, 16).expect("fills");
        for y in 4..12 {
            for x in 4..12 {
                assert_eq!(mask[y * 16 + x], 255, "({x}, {y}) should be solid");
            }
        }
        // Nothing outside the 8x8 block has any ink.
        assert_eq!(ink(&mask), 64.0, "exactly 64 pixels of ink");
    }

    #[test]
    fn half_pixel_offsets_give_exact_half_coverage() {
        // A rectangle from 4.5 to 8.5 horizontally: each edge column is half
        // covered, and analytically that is exactly 128.
        let mask = fill_path(&rect(4.5, 4.0, 4.0, 8.0), FillRule::NonZero, 16, 16).expect("fills");
        for y in 4..12 {
            assert_eq!(mask[y * 16 + 4], 128, "left edge at y={y}");
            assert_eq!(mask[y * 16 + 5], 255, "interior at y={y}");
            assert_eq!(mask[y * 16 + 6], 255, "interior at y={y}");
            assert_eq!(mask[y * 16 + 7], 255, "interior at y={y}");
            assert_eq!(mask[y * 16 + 8], 128, "right edge at y={y}");
        }
        // 4 pixels of ink per row: 0.5 + 3 + 0.5. The half edges are quantised
        // to 128, which is 0.50196 of a pixel rather than 0.5, so the total
        // carries 16 half-quantisation steps of slack.
        assert!((ink(&mask) - 32.0).abs() < 0.04, "ink {}", ink(&mask));
    }

    #[test]
    fn quarter_pixel_offsets_are_exact_too() {
        // 4.25 to 7.75 is 3.5 wide: columns 4 and 7 each hold three quarters, with
        // two solid columns between.
        let mask = fill_path(&rect(4.25, 0.0, 3.5, 4.0), FillRule::NonZero, 12, 4).expect("fills");
        assert_eq!(mask[4], 191, "three quarters of 255 rounds to 191");
        assert_eq!(mask[5], 255, "solid interior");
        assert_eq!(mask[6], 255, "solid interior");
        assert_eq!(mask[7], 191, "three quarters again");
        assert_eq!(mask[3], 0, "outside the left edge");
        assert_eq!(mask[8], 0, "outside the right edge");
        // 3.5 px of ink per row, 4 rows.
        assert!((ink(&mask) - 14.0).abs() < 0.02, "ink {}", ink(&mask));
    }

    #[test]
    fn nonzero_and_even_odd_differ_on_a_same_wound_donut() {
        let path = donut(40.0, 18.0);
        let nz = fill_path(&path, FillRule::NonZero, 100, 100).expect("fills");
        let eo = fill_path(&path, FillRule::EvenOdd, 100, 100).expect("fills");
        // Non-zero: the whole disc is ink (the two contours agree in
        // direction, so the winding never returns to zero).
        assert_eq!(nz[50 * 100 + 50], 255, "the centre is ink under non-zero");
        assert!(ink(&nz) > 1000.0, "non-zero ink {}", ink(&nz));
        // Even-odd: the inner contour punches a hole.
        assert_eq!(eo[50 * 100 + 50], 0, "the centre is a hole under even-odd");
        assert!(
            ink(&eo) < ink(&nz) * 0.9,
            "even-odd {} vs non-zero {}",
            ink(&eo),
            ink(&nz)
        );
        // The hole really is a hole: a ring of ink around an empty centre.
        assert_eq!(eo[50 * 100 + 50], 0, "the middle row's centre is a hole");
        // The ring at the outer edge is ink, though the flattened circle's
        // chord sits a hair inside the true circle, so it is 253 not 255.
        assert!(
            eo[50 * 100 + 10] > 250,
            "the ring around the hole is ink: {}",
            eo[50 * 100 + 10]
        );
        // Row 10 is above the inner circle but still inside the outer one: ink, and
        // the flattened chord sits a hair inside, so 254 rather than 255.
        assert!(
            eo[10 * 100 + 50] > 250,
            "above the hole: {}",
            eo[10 * 100 + 50]
        );
        // The very centre of the raster is deep inside the ring on both rules.
        assert_eq!(nz[50 * 100 + 50], 255, "non-zero fills the middle");
        assert_eq!(eo[50 * 100 + 50], 0, "even-odd punches it out");
    }

    #[test]
    fn opposite_wound_donut_is_a_hole_under_both_rules() {
        // An outer circle one way, an inner circle the other: the classic
        // counter, where both rules agree the middle is a hole.
        //
        // `Path2D::reverse` flips *every* subpath, so the inner contour is
        // built and reversed on its own before being appended.
        let mut outer_builder = PathBuilder::new();
        outer_builder.circle(50.0, 50.0, 40.0);
        let outer = outer_builder.build();

        let mut inner_builder = PathBuilder::new();
        inner_builder.circle(50.0, 50.0, 18.0);
        let mut inner = inner_builder.build();
        inner.reverse();

        let mut b = PathBuilder::new();
        b.extend(&outer);
        b.extend(&inner);
        let path = b.build();

        for rule in [FillRule::NonZero, FillRule::EvenOdd] {
            let mask = fill_path(&path, rule, 100, 100).expect("fills");
            assert_eq!(mask[50 * 100 + 50], 0, "hole under {rule:?}");
            assert!(
                mask[50 * 100 + 10] > 250,
                "the ring is ink under {rule:?}: {}",
                mask[50 * 100 + 10]
            );
            // A point well inside the ring, clear of the flattened circle's
            // edge, is solid.
            assert!(
                mask[10 * 100 + 50] > 250,
                "solid ring under {rule:?}: {}",
                mask[10 * 100 + 50]
            );
        }
    }

    #[test]
    fn self_intersecting_star_differs_between_the_rules() {
        // A five-pointed star drawn as one self-intersecting contour: the
        // middle pentagon is wound twice, so non-zero fills it and even-odd
        // punches it out. This is the case that separates the rules.
        let pts = [
            (50.0, 10.0),
            (19.0, 90.0),
            (90.0, 38.0),
            (10.0, 38.0),
            (81.0, 90.0),
        ];
        let mut b = PathBuilder::new();
        for (i, p) in pts.iter().enumerate() {
            if i == 0 {
                b.move_to(p.0, p.1);
            } else {
                b.line_to(p.0, p.1);
            }
        }
        b.close();
        let path = b.build();

        let nz = fill_path(&path, FillRule::NonZero, 100, 100).expect("fills");
        let eo = fill_path(&path, FillRule::EvenOdd, 100, 100).expect("fills");
        // The doubly-wound centre: ink under non-zero, a hole under even-odd.
        assert_eq!(nz[50 * 100 + 50], 255, "non-zero keeps the centre");
        assert_eq!(eo[50 * 100 + 50], 0, "even-odd empties the centre");
        // Both rules agree on a point in a single-wound arm.
        assert_eq!(nz[32 * 100 + 50], 255, "single-wound arm");
        assert_eq!(eo[32 * 100 + 50], 255, "single-wound arm");
        // And the empty corner outside the star.
        assert_eq!(nz[50 * 100 + 5], 0);
        assert_eq!(eo[50 * 100 + 5], 0);
        // The two rules really do differ by the pentagon.
        assert!(
            ink(&nz) - ink(&eo) > 100.0,
            "non-zero {} vs even-odd {}",
            ink(&nz),
            ink(&eo)
        );
    }

    #[test]
    fn degenerate_paths_produce_no_panic_and_no_ink() {
        let empty = crate::Path2D::new();
        let mask = fill_path(&empty, FillRule::NonZero, 8, 8).expect("fills");
        assert!(mask.iter().all(|&v| v == 0));

        // A lone point.
        let mut b = PathBuilder::new();
        b.move_to(4.0, 4.0);
        let mask = fill_path(&b.build(), FillRule::NonZero, 8, 8).expect("fills");
        assert_eq!(ink(&mask), 0.0);

        // A zero-length line.
        let mut b = PathBuilder::new();
        b.move_to(4.0, 4.0);
        b.line_to(4.0, 4.0);
        b.close();
        let mask = fill_path(&b.build(), FillRule::NonZero, 8, 8).expect("fills");
        assert_eq!(ink(&mask), 0.0);

        // A zero-area contour: a degenerate polygon.
        let mut b = PathBuilder::new();
        b.move_to(1.0, 1.0);
        b.line_to(5.0, 1.0);
        b.line_to(3.0, 1.0);
        b.close();
        let mask = fill_path(&b.build(), FillRule::NonZero, 8, 8).expect("fills");
        assert!(
            ink(&mask) < 0.01,
            "a flat contour has no area: {}",
            ink(&mask)
        );

        // A self-cancelling outline: a shape drawn forwards then backwards.
        let mut b = PathBuilder::new();
        b.rect(1.0, 1.0, 4.0, 4.0);
        b.rect(1.0, 1.0, 4.0, 4.0);
        let mask = fill_path(&b.build(), FillRule::NonZero, 8, 8).expect("fills");
        assert!((ink(&mask) - 16.0).abs() < 0.01, "overlap still fills");
    }

    #[test]
    fn triangle_coverage_sums_to_its_area() {
        // A right triangle with legs 64: area = 2048. The mask's total ink must
        // match within 0.1 %, which is the bound the analytic filler earns.
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0);
        b.line_to(64.0, 0.0);
        b.line_to(0.0, 64.0);
        b.close();
        let path = b.build();
        let mask = fill_path(&path, FillRule::NonZero, 64, 64).expect("fills");
        let total = ink(&mask);
        let exact = 0.5 * 64.0 * 64.0;
        let err = (total - exact).abs() / exact;
        assert!(
            err < 0.001,
            "triangle ink {total} vs area {exact}: relative error {err}"
        );
        // A tilted triangle: the same bound with diagonal edges, where a
        // supersampling rasteriser would drift.
        let mut t = PathBuilder::new();
        t.move_to(10.0, 10.0);
        t.line_to(90.0, 20.0);
        t.line_to(30.0, 90.0);
        t.close();
        let mask = fill_path(&t.build(), FillRule::NonZero, 100, 100).expect("fills");
        // Shoelace of the exact triangle, summed in `f64` from `f32` vertices.
        let pts = [(10.0f32, 10.0f32), (90.0, 20.0), (30.0, 90.0)];
        let mut acc = 0.0f64;
        for (i, a) in pts.iter().enumerate() {
            let b = pts[(i + 1) % 3];
            acc += f64::from(a.0) * f64::from(b.1) - f64::from(b.0) * f64::from(a.1);
        }
        let exact = (acc * 0.5).abs();
        let err = (ink(&mask) - exact).abs() / exact;
        assert!(err < 0.002, "tilted triangle error {err}");
    }

    #[test]
    fn circle_coverage_sums_to_its_area() {
        // A radius-40 circle: area = pi·1600 ≈ 5026.5. The flattening tolerance
        // bounds the error, so 1 % is generous.
        let mut b = PathBuilder::new();
        b.circle(50.0, 50.0, 40.0);
        let mask = fill_path(&b.build(), FillRule::NonZero, 100, 100).expect("fills");
        let want = core::f64::consts::PI * 1600.0;
        let err = (ink(&mask) - want).abs() / want;
        assert!(err < 0.01, "circle ink error {err}");
    }

    #[test]
    fn zero_sized_raster_is_empty() {
        let p = rect(0.0, 0.0, 4.0, 4.0);
        assert!(fill_path(&p, FillRule::NonZero, 0, 8)
            .expect("fills")
            .is_empty());
        assert!(fill_path(&p, FillRule::NonZero, 8, 0)
            .expect("fills")
            .is_empty());
        assert!(fill_path(&p, FillRule::NonZero, 0, 0)
            .expect("fills")
            .is_empty());
    }

    #[test]
    fn oversized_raster_is_refused() {
        let p = rect(0.0, 0.0, 1.0, 1.0);
        match fill_path(&p, FillRule::NonZero, 100_000, 100_000).unwrap_err() {
            ShapeError::RasterTooLarge { pixels, limit } => {
                assert_eq!(pixels, 10_000_000_000);
                assert_eq!(limit, MAX_MASK_ELEMENTS);
            }
            other => panic!("expected RasterTooLarge, got {other:?}"),
        }
        // And the message names the numbers.
        let msg = alloc::format!(
            "{}",
            ShapeError::RasterTooLarge {
                pixels: 10_000_000_000,
                limit: MAX_MASK_ELEMENTS
            }
        );
        assert!(msg.contains("10000000000"));
    }

    #[test]
    fn geometry_entirely_outside_the_raster_is_empty() {
        let p = rect(100.0, 100.0, 10.0, 10.0);
        let mask = fill_path(&p, FillRule::NonZero, 8, 8).expect("fills");
        assert_eq!(ink(&mask), 0.0);
        // And partly outside is clipped, not wrapped.
        let p = rect(-4.0, -4.0, 8.0, 8.0);
        let mask = fill_path(&p, FillRule::NonZero, 8, 8).expect("fills");
        assert!((ink(&mask) - 16.0).abs() < 0.01, "clipped to a quarter");
    }

    #[test]
    fn collect_edges_drops_horizontal_and_non_finite_segments() {
        // A horizontal-only path has no edges at all.
        let mut b = PathBuilder::new();
        b.move_to(0.0, 5.0);
        b.line_to(10.0, 5.0);
        b.close();
        assert!(collect_edges(&b.build()).is_empty());

        // A NaN segment is dropped, and the good one survives.
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0);
        b.line_to(f32::NAN, 10.0);
        b.line_to(10.0, 20.0);
        b.close();
        let edges = collect_edges(&b.build());
        assert_eq!(edges.len(), 1, "one usable edge");
        assert!(edges[0].x0.is_finite() && edges[0].x1.is_finite());
    }

    #[test]
    fn edges_report_their_direction() {
        let down = Edge {
            x0: 0.0,
            y0: 0.0,
            x1: 1.0,
            y1: 4.0,
            dir: 1,
        };
        assert_eq!(down.y_min(), 0.0);
        assert_eq!(down.y_max(), 4.0);
        assert!(close(down.x_at(0.0), 0.0));
        assert!(close(down.x_at(2.0), 0.5));
        assert!(close(down.x_at(4.0), 1.0));
        // A degenerate edge with no vertical extent evaluates to x0.
        let flat = Edge {
            x0: 3.0,
            y0: 1.0,
            x1: 3.0,
            y1: 1.0,
            dir: 1,
        };
        assert!(close(flat.x_at(1.0), 3.0));
    }

    #[test]
    fn fill_rule_default_is_non_zero() {
        assert_eq!(FillRule::default(), FillRule::NonZero);
    }

    #[test]
    fn transform_of_a_path_matches_its_mask_bounds() {
        // Fill a rectangle, then a transformed copy: the ink is the same
        // because a transform of a path is a transform of its coverage.
        let p = rect(20.0, 30.0, 20.0, 10.0);
        let a = fill_path(&p, FillRule::NonZero, 64, 64).expect("fills");
        let t = p.transformed(&crate::Affine2D::translate(4.0, -6.0));
        let b = fill_path(&t, FillRule::NonZero, 64, 64).expect("fills");
        assert!(
            (ink(&a) - ink(&b)).abs() < 0.5,
            "{} vs {}",
            ink(&a),
            ink(&b)
        );
        // A uniform scale by 2 quadruples the area.
        let big = p.transformed(&crate::Affine2D::scale(2.0, 2.0));
        let c = fill_path(&big, FillRule::NonZero, 128, 128).expect("fills");
        assert!(
            (ink(&c) - ink(&a) * 4.0).abs() < 1.0,
            "{} vs {}",
            ink(&c),
            ink(&a) * 4.0
        );
    }

    #[test]
    fn pixel_bounds_of_a_path() {
        let p = rect(4.25, 6.75, 10.0, 5.0);
        assert_eq!(path_pixel_bounds(&p, 64, 64), Some((4, 6, 11, 6)));
        // Clipped to the raster.
        assert_eq!(path_pixel_bounds(&p, 8, 8), Some((4, 6, 4, 2)));
        // Nothing at all when off-raster.
        assert_eq!(path_pixel_bounds(&rect(100.0, 100.0, 1.0, 1.0), 8, 8), None);
        // And an empty path has no bounds.
        assert_eq!(path_pixel_bounds(&crate::Path2D::new(), 8, 8), None);
    }

    #[test]
    fn mask_values_are_always_in_range() {
        // A wild path: a star, several overlapping shapes, and a curve.
        let mut b = PathBuilder::new();
        b.circle(32.0, 32.0, 30.0);
        for i in 0..7 {
            let a = i as f32 * 0.9;
            b.move_to(32.0, 32.0);
            b.line_to(32.0 + 25.0 * libm::cosf(a), 32.0 + 25.0 * libm::sinf(a));
        }
        b.rect(0.0, 0.0, 64.0, 64.0);
        for rule in [FillRule::NonZero, FillRule::EvenOdd] {
            let mask = fill_path(&b.build(), rule, 64, 64).expect("fills");
            assert_eq!(mask.len(), 64 * 64);
            // Every pixel is a valid coverage byte — the type says so — and
            // the total ink cannot exceed the frame it covers.
            assert!(ink(&mask) <= 64.0 * 64.0 + 1e-6);
            assert!(ink(&mask) > 0.0, "the star and the frame both draw");
        }
    }
}
