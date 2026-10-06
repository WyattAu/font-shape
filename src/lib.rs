//! Glyph rasterization and text measurement — scanline fill, stroke expansion,
//! advance/kerning layout, and bidirectional reordering.
//!
//! `font-shape` is the estate's **L2 shaping layer**: the half of a font
//! toolchain that turns a [`font_model::Font`] into pixels and into measured
//! text. It sits on [`font-model`] (L1), which sits on [`font-parse`] (L0).
//!
//! ```
//! use font_model::{Font, Glyph, GlyphId, Outline, Path};
//! use font_shape::{rasterize_glyph, shape_line, Hinting};
//!
//! // A 1000-upem font with one square glyph, 500 units wide.
//! let mut outline = Outline::new();
//! let mut p = Path::starting_at(50.0, 0.0);
//! p.line_to(450.0, 0.0);
//! p.line_to(450.0, 700.0);
//! p.line_to(50.0, 700.0);
//! p.close();
//! outline.push_contour(p);
//!
//! let font = Font::builder()
//!     .units_per_em(1000)
//!     .glyph(Glyph::empty(GlyphId::NOTDEF, 500))
//!     .glyph(Glyph::new(GlyphId::new(1), 500, 50, outline))
//!     .build()
//!     .expect("a one-glyph font is valid");
//!
//! // Rasterize at 20 px: an 8x14 block, 20 rows above the baseline.
//! let bmp = rasterize_glyph(&font, GlyphId::new(1), 20.0, Hinting::None)?;
//! assert_eq!((bmp.width(), bmp.height()), (8, 14));
//! assert_eq!(bmp.top(), 14);
//! assert!(bmp.ink() > 100.0);
//!
//! // Measure the run: one glyph, 500 units at 20 px = 10 px.
//! let line = shape_line(&font, "A", 20.0)?;
//! assert_eq!(line.len(), 1);
//! assert!((line.advance_px() - 10.0).abs() < 1e-4);
//! # Ok::<(), font_shape::ShapeError>(())
//! ```
//!
//! # What is in here
//!
//! | Area | Entry points |
//! |---|---|
//! | transforms | [`Affine2D`] |
//! | paths | [`Path2D`], [`PathBuilder`] with a transform stack |
//! | filling | [`fill_path`] — exact analytic coverage, [`FillRule`] |
//! | stroking | [`stroke_path`] — expand to an outline, then fill |
//! | glyphs | [`rasterize_glyph`], [`rasterize_outline`], [`Hinting`], [`GlyphBitmap`] |
//! | text | [`shape_line`], [`measure_run`], [`measure_glyph`], [`resolve_bidi`] |
//!
//! # The rasterizer
//!
//! [`fill_path`] computes the **exact area** each pixel covers, integrating the
//! span coverage analytically rather than supersampling. Two properties follow,
//! and both are asserted in the test suite:
//!
//! - An axis-aligned rectangle on integer boundaries is *exactly* `255` in its
//!   interior and `0` outside; shifted half a pixel it is *exactly* `128` on
//!   each edge.
//! - A mask's total coverage equals the polygon it came from: for a 64-unit
//!   right triangle, the ink matches `½·64·64` to within 0.1 %. That single
//!   number bounds every antialiasing claim this crate makes.
//!
//! Device space is **y down**, pixel `(x, y)` covering `[x, x+1) × [y, y+1)`.
//! [`rasterize_glyph`] flips a font's y-up outline into that space and carries
//! all intermediate arithmetic in 26.6 fixed point, so a glyph lands on the
//! pixel grid the same way at 12 px and 144 px.
//!
//! ```
//! use font_shape::{fill_path, FillRule, PathBuilder};
//!
//! let mut b = PathBuilder::new();
//! b.rect(0.5, 0.0, 4.0, 1.0);
//! let mask = fill_path(&b.build(), FillRule::NonZero, 8, 1)?;
//!
//! assert_eq!(mask[0], 128, "half a pixel of coverage is exactly 128");
//! assert_eq!(mask[1], 255);
//! assert_eq!(mask[4], 128);
//! assert_eq!(mask[5], 0);
//! # Ok::<(), font_shape::ShapeError>(())
//! ```
//!
//! # Layer
//!
//! L2 — domain. Its only estate-internal dependency is `font-model` (L1).
//! `libm` supplies the `no_std` transcendentals and `thiserror` the error
//! taxonomy. There is deliberately **no** image or rasterizer dependency: the
//! scanline filler is the point of the crate.
//!
//! [`font-parse`]: https://docs.rs/font-parse

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(missing_docs)]
#![forbid(unsafe_code)]

extern crate alloc;

mod affine;
mod error;
mod glyph;
mod measure;
mod path;
mod raster;
mod stroke;

pub use affine::Affine2D;
pub use error::{message, Result, ShapeError};
pub use glyph::{
    em_size_px, fixed_scale, from_fixed, glyph_path, glyph_path_from_outline, grid_fit,
    rasterize_glyph, rasterize_outline, scaled_metrics, to_26_6, to_fixed, DeviceMetrics,
    GlyphBitmap, Hinting, PIXEL, PIXEL_FP_BITS,
};
pub use measure::{
    ascii_cmap, bidi_class, describe_run, is_rtl_level, measure_glyph, measure_run,
    reorder_indices, resolve_bidi, resolve_bidi_chars, set_explicit_levels, shape_line,
    shape_line_with, BidiClass, BidiLevels, GlyphPlacement, LayoutDirection, ShapedLine,
};
pub use path::{Path2D, PathBuilder, PathCommand, Rect};
pub use raster::{fill_path, mask_coverage, path_pixel_bounds, FillRule, MAX_MASK_ELEMENTS};
pub use stroke::{stroke_path, LineCap, LineJoin, StrokeStyle, DEFAULT_MITER_LIMIT};

/// The crate's version, for callers that report it.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
