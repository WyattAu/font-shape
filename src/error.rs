//! Typed, exhaustive error taxonomy for the rasteriser and the measurement
//! layer.
//!
//! The contract under fuzz and hostile geometry is *totality with a typed
//! escape hatch*: a singular transform, an empty raster, or a missing glyph is
//! one of these variants — never a panic, never an out-of-bounds write. The
//! enum is deliberately exhaustive (no `#[non_exhaustive]`): callers match every
//! variant, and a new variant is a semver-minor event by design.

use alloc::format;
use alloc::string::String;
use core::fmt;

/// The single crate-level error for rasterization and measurement.
#[derive(Debug, Clone, PartialEq)]
pub enum ShapeError {
    /// The affine transform's linear part is singular, so it has no inverse
    /// and collapses the plane.
    SingularAffine {
        /// The determinant that was (near) zero.
        determinant: f32,
    },
    /// A transform stack was popped when empty. [`PathBuilder::pop_transform`]
    /// returns this instead of quietly returning the identity, which would
    /// silently drop a transform.
    ///
    /// [`PathBuilder::pop_transform`]: crate::PathBuilder::pop_transform
    UnbalancedPop {
        /// The stack depth at the pop.
        depth: usize,
    },
    /// A size argument was zero, negative, or otherwise outside the usable
    /// range for a raster of this dimension.
    InvalidSize {
        /// The requested width in pixels.
        width: u32,
        /// The requested height in pixels.
        height: u32,
    },
    /// A scale factor was zero, negative, NaN, or infinite.
    InvalidScale {
        /// The rejected scale.
        scale: f32,
    },
    /// A glyph id beyond the font's store was requested.
    UnknownGlyph {
        /// The dangling id's numeric value.
        glyph: u32,
        /// How many glyphs the font holds.
        glyph_count: usize,
    },
    /// A coordinate was NaN or infinite where a finite one was required.
    NonFiniteCoordinate {
        /// Which axis: `"x"` or `"y"`.
        axis: &'static str,
    },
    /// The raster is too large to allocate: `width × height` exceeded the
    /// element cap, so the mask would be gigabytes.
    RasterTooLarge {
        /// Total mask elements requested.
        pixels: usize,
        /// The cap that was exceeded.
        limit: usize,
    },
    /// A font's `units_per_em` was zero, which would make the scale factor
    /// infinite.
    DegenerateUnitsPerEm,
}

impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShapeError::SingularAffine { determinant } => write!(
                f,
                "affine transform is singular (determinant {determinant})"
            ),
            ShapeError::UnbalancedPop { depth } => {
                write!(f, "transform stack popped at depth {depth} (already empty)")
            }
            ShapeError::InvalidSize { width, height } => {
                write!(f, "invalid raster size {width}x{height}")
            }
            ShapeError::InvalidScale { scale } => write!(f, "invalid scale {scale}"),
            ShapeError::UnknownGlyph { glyph, glyph_count } => write!(
                f,
                "glyph {} requested but the font holds only {glyph_count}",
                glyph
            ),
            ShapeError::NonFiniteCoordinate { axis } => {
                write!(f, "{axis} coordinate is not finite")
            }
            ShapeError::RasterTooLarge { pixels, limit } => {
                write!(f, "raster of {pixels} pixels exceeds the {limit}-pixel cap")
            }
            ShapeError::DegenerateUnitsPerEm => write!(f, "font has unitsPerEm == 0"),
        }
    }
}

impl From<crate::affine::Affine2D> for ShapeError {
    /// A transform with no inverse is the only way this conversion applies.
    fn from(_t: crate::affine::Affine2D) -> Self {
        ShapeError::SingularAffine { determinant: 0.0 }
    }
}

impl From<core::num::ParseFloatError> for ShapeError {
    /// A string that was meant to be a scale did not parse.
    fn from(_e: core::num::ParseFloatError) -> Self {
        ShapeError::InvalidScale { scale: f32::NAN }
    }
}

/// A convenience alias for this crate's error.
pub type Result<T> = core::result::Result<T, ShapeError>;

/// The message a `ShapeError` carries, for diagnostics in a host that cannot
/// propagate the type.
#[must_use]
pub fn message(e: &ShapeError) -> String {
    format!("{e}")
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::float_cmp
    )]
    use super::message;
    use crate::ShapeError;
    use alloc::string::ToString;

    // Test harness: assertions legitimately panic; the lib target stays
    // lint-clean.

    #[test]
    fn every_variant_renders_informatively() {
        let cases = [
            (ShapeError::SingularAffine { determinant: 0.0 }, "singular"),
            (ShapeError::UnbalancedPop { depth: 0 }, "popped"),
            (
                ShapeError::InvalidSize {
                    width: 0,
                    height: 16,
                },
                "0x16",
            ),
            (ShapeError::InvalidScale { scale: -1.0 }, "-1"),
            (
                ShapeError::UnknownGlyph {
                    glyph: 9,
                    glyph_count: 2,
                },
                "glyph 9",
            ),
            (
                ShapeError::NonFiniteCoordinate { axis: "y" },
                "y coordinate",
            ),
            (
                ShapeError::RasterTooLarge {
                    pixels: 1_000_000,
                    limit: 16_777_216,
                },
                "1000000",
            ),
            (ShapeError::DegenerateUnitsPerEm, "unitsPerEm == 0"),
        ];
        for (e, needle) in cases {
            let s = e.to_string();
            assert!(!s.is_empty());
            assert!(s.contains(needle), "{s:?} should mention {needle:?}");
            assert_eq!(message(&e), s);
        }
    }

    #[test]
    fn error_is_send_sync_and_static() {
        // A `'static + Send + Sync` error composes into any host's error type.
        fn assert_bounds<T: Send + Sync + 'static>(_: &T) {}
        assert_bounds(&ShapeError::DegenerateUnitsPerEm);
        // And it is Clone, so a host can keep a copy for its diagnostics.
        let e = ShapeError::DegenerateUnitsPerEm;
        assert_eq!(e.clone(), e);
    }
}
