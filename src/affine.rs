//! 2×3 affine transforms.
//!
//! A matrix here is the six meaningful values of a 2D affine map, stored in
//! the order a font's composite glyph transforms use:
//!
//! ```text
//!     | a  c  e |     x' = a·x + c·y + e
//!     | b  d  f |     y' = b·x + d·y + f
//!     | 0  0  1 |
//! ```
//!
//! `then` composes in application order: `a.then(&b)` is the transform that
//! applies `a` and *then* `b`, so `let t = scale.then(&translate)` translates
//! by the scaled amount. That is the composition a nested
//! `push_transform`/`pop_transform` stack wants, and the one the
//! [`PathBuilder`](crate::PathBuilder) applies.
//!
//! ```
//! use font_shape::Affine2D;
//!
//! // Scale by 2, then translate by 10: the translate is *inside* the scale.
//! let t = Affine2D::scale(2.0, 2.0).then(&Affine2D::translate(10.0, 0.0));
//! assert_eq!(t.transform_point(1.0, 0.0), (12.0, 0.0));
//!
//! // Inversion round-trips.
//! let inv = t.inverse().expect("scale-then-translate is invertible");
//! let p = t.transform_point(3.0, 4.0);
//! assert_eq!(inv.transform_point(p.0, p.1), (3.0, 4.0));
//! ```

use crate::error::ShapeError;

/// A 2D affine transform: scale, rotate, skew, then translate.
///
/// ```
/// use font_shape::Affine2D;
///
/// let r = Affine2D::rotate(core::f32::consts::FRAC_PI_2);
/// // A quarter turn sends (1, 0) to (0, 1).
/// let (x, y) = r.transform_point(1.0, 0.0);
/// assert!((x).abs() < 1e-6 && (y - 1.0).abs() < 1e-6, "({x}, {y})");
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Affine2D {
    /// Row 0, column 0: the x scale.
    pub a: f32,
    /// Row 1, column 0: the x shear.
    pub b: f32,
    /// Row 0, column 1: the y shear.
    pub c: f32,
    /// Row 1, column 1: the y scale.
    pub d: f32,
    /// Row 0, column 2: the x translation.
    pub e: f32,
    /// Row 1, column 2: the y translation.
    pub f: f32,
}

impl Default for Affine2D {
    fn default() -> Self {
        Affine2D::IDENTITY
    }
}

impl Affine2D {
    /// The identity transform.
    pub const IDENTITY: Affine2D = Affine2D {
        a: 1.0,
        b: 0.0,
        c: 0.0,
        d: 1.0,
        e: 0.0,
        f: 0.0,
    };

    /// A transform from its six values, in `(a, b, c, d, e, f)` order.
    #[must_use]
    pub const fn new(a: f32, b: f32, c: f32, d: f32, e: f32, f: f32) -> Self {
        Affine2D { a, b, c, d, e, f }
    }

    /// A translation.
    #[must_use]
    pub const fn translate(dx: f32, dy: f32) -> Self {
        Affine2D::new(1.0, 0.0, 0.0, 1.0, dx, dy)
    }

    /// A non-uniform (or uniform) scale.
    #[must_use]
    pub const fn scale(sx: f32, sy: f32) -> Self {
        Affine2D::new(sx, 0.0, 0.0, sy, 0.0, 0.0)
    }

    /// A rotation by `radians`, clockwise in a y-down coordinate system
    /// (which is the space a rasteriser works in).
    #[must_use]
    pub fn rotate(radians: f32) -> Self {
        let (s, c) = (libm::sinf(radians), libm::cosf(radians));
        Affine2D::new(c, s, -s, c, 0.0, 0.0)
    }

    /// A rotation by `degrees`.
    #[must_use]
    pub fn rotate_degrees(degrees: f32) -> Self {
        Self::rotate(degrees * core::f32::consts::PI / 180.0)
    }

    /// A skew along x by `radians`.
    #[must_use]
    pub fn skew_x(radians: f32) -> Self {
        Affine2D::new(1.0, 0.0, libm::tanf(radians), 1.0, 0.0, 0.0)
    }

    /// A skew along y by `radians`.
    #[must_use]
    pub fn skew_y(radians: f32) -> Self {
        Affine2D::new(1.0, libm::tanf(radians), 0.0, 1.0, 0.0, 0.0)
    }

    /// True when this is the identity to within a small tolerance.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        let t = self;
        (t.a - 1.0).abs() < 1e-6
            && t.b.abs() < 1e-6
            && t.c.abs() < 1e-6
            && (t.d - 1.0).abs() < 1e-6
            && t.e.abs() < 1e-6
            && t.f.abs() < 1e-6
    }

    /// Compose: `self.then(&next)` applies `self` and *then* `next`.
    ///
    /// ```
    /// use font_shape::Affine2D;
    ///
    /// let t = Affine2D::translate(1.0, 0.0).then(&Affine2D::scale(2.0, 2.0));
    /// // Translate by 1, then double: (1, 0) -> (2, 0) -> (4, 0).
    /// assert_eq!(t.transform_point(1.0, 0.0), (4.0, 0.0));
    /// ```
    #[must_use]
    pub fn then(&self, next: &Affine2D) -> Affine2D {
        Affine2D::new(
            next.a * self.a + next.c * self.b,
            next.b * self.a + next.d * self.b,
            next.a * self.c + next.c * self.d,
            next.b * self.c + next.d * self.d,
            next.a * self.e + next.c * self.f + next.e,
            next.b * self.e + next.d * self.f + next.f,
        )
    }

    /// The determinant of the linear part, `a·d − b·c`.
    #[must_use]
    pub fn determinant(&self) -> f32 {
        self.a * self.d - self.b * self.c
    }

    /// True when the linear part is singular — the transform collapses the
    /// plane and has no inverse.
    #[must_use]
    pub fn is_singular(&self) -> bool {
        self.determinant().abs() <= 1e-12
    }

    /// The inverse transform.
    ///
    /// # Errors
    /// [`ShapeError::SingularAffine`] when the determinant is (near) zero, so
    /// the caller gets a typed error rather than a matrix full of infinities.
    #[allow(clippy::result_large_err)]
    pub fn inverse(&self) -> Result<Affine2D, ShapeError> {
        let det = self.determinant();
        if det.abs() <= 1e-12 {
            return Err(ShapeError::SingularAffine { determinant: det });
        }
        let inv = 1.0 / det;
        Ok(Affine2D::new(
            self.d * inv,
            -self.b * inv,
            -self.c * inv,
            self.a * inv,
            (self.c * self.f - self.d * self.e) * inv,
            (self.b * self.e - self.a * self.f) * inv,
        ))
    }

    /// Transform a point: the translation applies.
    #[must_use]
    pub fn transform_point(&self, x: f32, y: f32) -> (f32, f32) {
        (
            self.a * x + self.c * y + self.e,
            self.b * x + self.d * y + self.f,
        )
    }

    /// Transform a vector: the translation does *not* apply.
    #[must_use]
    pub fn transform_vector(&self, x: f32, y: f32) -> (f32, f32) {
        (self.a * x + self.c * y, self.b * x + self.d * y)
    }

    /// The transform's x-axis mapped to a unit vector: where a unit step in x
    /// lands. Used by the stroker to find edge normals.
    #[must_use]
    pub fn x_axis(&self) -> (f32, f32) {
        (self.a, self.b)
    }

    /// The transform's y-axis mapped to a unit vector.
    #[must_use]
    pub fn y_axis(&self) -> (f32, f32) {
        (self.c, self.d)
    }

    /// The average scale factor — the cube root of the absolute determinant.
    ///
    /// A transform's "size", for scaling a stroke width by something that does
    /// not depend on which way it was rotated. A reflection gives the same
    /// positive value as the transform it mirrors.
    #[must_use]
    pub fn scale_factor(&self) -> f32 {
        libm::cbrtf(self.determinant().abs())
    }
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
    use super::Affine2D;
    use crate::ShapeError;
    use core::f32::consts::E;
    use core::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    // Test harness: assertions legitimately panic; the lib target stays
    // lint-clean.

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn identity_is_the_neutral_element() {
        let id = Affine2D::IDENTITY;
        assert!(id.is_identity());
        assert_eq!(id.determinant(), 1.0);
        assert_eq!(id.transform_point(3.0, 4.0), (3.0, 4.0));
        assert_eq!(id.transform_vector(3.0, 4.0), (3.0, 4.0));
        assert!(id.inverse().expect("invertible").is_identity());
        assert_eq!(Affine2D::default(), id);
    }

    #[test]
    fn composition_order_is_application_order() {
        // translate then scale: (1,0) -> (2,0) -> (4,0)
        let t = Affine2D::translate(1.0, 0.0).then(&Affine2D::scale(2.0, 2.0));
        assert_eq!(t.transform_point(1.0, 0.0), (4.0, 0.0));
        // scale then translate: (1,0) -> (2,0) -> (3,0)
        let s = Affine2D::scale(2.0, 2.0).then(&Affine2D::translate(1.0, 0.0));
        assert_eq!(s.transform_point(1.0, 0.0), (3.0, 0.0));
        // Composition is associative.
        let a = Affine2D::scale(3.0, 1.0);
        let b = Affine2D::rotate(0.7);
        let c = Affine2D::translate(-4.0, 2.0);
        let left = a.then(&b).then(&c);
        let right = a.then(&b.then(&c));
        let p = (1.5, -2.5);
        assert!(close(
            left.transform_point(p.0, p.1).0,
            right.transform_point(p.0, p.1).0
        ));
        assert!(close(
            left.transform_point(p.0, p.1).1,
            right.transform_point(p.0, p.1).1
        ));
    }

    #[test]
    fn inverse_round_trips_points_and_vectors() {
        let cases = [
            Affine2D::new(2.0, 0.5, -0.25, 3.0, 7.0, -8.0),
            Affine2D::rotate(0.9).then(&Affine2D::translate(3.0, 4.0)),
            Affine2D::scale(-1.0, 2.0).then(&Affine2D::translate(1.0, -1.0)),
        ];
        for t in cases {
            let inv = t.inverse().expect("invertible");
            let p = t.transform_point(2.5, -7.25);
            let back = inv.transform_point(p.0, p.1);
            assert!(close(back.0, 2.5), "{t:?} point x {}", back.0);
            assert!(close(back.1, -7.25), "{t:?} point y {}", back.1);
            let v = t.transform_vector(2.5, -7.25);
            let vback = inv.transform_vector(v.0, v.1);
            assert!(close(vback.0, 2.5));
            assert!(close(vback.1, -7.25));
            // The inverse's inverse is the original.
            let again = inv.inverse().expect("invertible");
            assert!(close(again.a, t.a) && close(again.d, t.d) && close(again.e, t.e));
        }
    }

    #[test]
    fn singular_matrices_are_rejected() {
        for t in [
            Affine2D::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            Affine2D::new(1.0, 2.0, 2.0, 4.0, 5.0, 6.0),
            Affine2D::scale(0.0, 1.0),
            Affine2D::scale(1.0, 0.0),
        ] {
            assert!(t.is_singular(), "{t:?} should be singular");
            assert_eq!(t.determinant(), 0.0);
            match t.inverse().unwrap_err() {
                ShapeError::SingularAffine { determinant } => {
                    assert!(determinant.abs() < 1e-12);
                }
                other => panic!("expected SingularAffine, got {other:?}"),
            }
        }
        // A near-singular matrix is refused rather than returning infinities.
        let near = Affine2D::new(1e-9, 0.0, 0.0, 1e-9, 0.0, 0.0);
        assert!(near.is_singular());
        assert!(near.inverse().is_err());
    }

    #[test]
    fn point_and_vector_differ_only_by_translation() {
        // A pure translation: the only difference between a point and a vector is
        // the offset, unscaled by anything.
        let t = Affine2D::translate(100.0, -50.0);
        let p = t.transform_point(1.0, 1.0);
        let v = t.transform_vector(1.0, 1.0);
        assert_eq!((p.0 - v.0, p.1 - v.1), (100.0, -50.0));

        // With a scale composed on top, the translation is inside the scale,
        // so it is scaled too: scale-then-translate moves the origin by
        // twice its offset in x.
        let s = Affine2D::translate(100.0, -50.0).then(&Affine2D::scale(2.0, 3.0));
        let p = s.transform_point(0.0, 0.0);
        assert!(
            close(p.0, 200.0) && close(p.1, -150.0),
            "({}, {})",
            p.0,
            p.1
        );
        assert_eq!(s.transform_vector(1.0, 1.0), (2.0, 3.0));
        assert!(s.transform_vector(1.0, 1.0) != s.transform_point(1.0, 1.0));
    }

    #[test]
    fn rotate_by_a_quarter_turn() {
        let r = Affine2D::rotate(FRAC_PI_2);
        let (x, y) = r.transform_point(1.0, 0.0);
        assert!(close(x, 0.0) && close(y, 1.0), "({x}, {y})");
        // Four quarter turns is the identity.
        let mut t = Affine2D::IDENTITY;
        for _ in 0..4 {
            t = t.then(&Affine2D::rotate(FRAC_PI_2));
        }
        assert!(t.is_identity(), "{t:?}");
        // And a half turn negates.
        let h = Affine2D::rotate(PI);
        assert!(close(h.transform_point(3.0, 4.0).0, -3.0));
        assert!(close(h.transform_point(3.0, 4.0).1, -4.0));
    }

    #[test]
    fn rotate_degrees_matches_radians() {
        let a = Affine2D::rotate_degrees(90.0);
        let b = Affine2D::rotate(FRAC_PI_2);
        assert!(close(a.a, b.a) && close(a.b, b.b));
        assert!(close(Affine2D::rotate_degrees(180.0).a, -1.0));
    }

    #[test]
    fn skews_shear_along_one_axis_only() {
        let sx = Affine2D::skew_x(FRAC_PI_4);
        let (x, y) = sx.transform_point(0.0, 1.0);
        assert!(close(x, 1.0), "skew_x leaves x alone: {x}");
        assert!(close(y, 1.0), "skew_x leaves y alone: {y}");
        let (x, _) = sx.transform_point(1.0, 1.0);
        assert!(close(x, 2.0), "skew_x adds tan(45°)·y to x: {x}");
        let sy = Affine2D::skew_y(FRAC_PI_4);
        let (x, y) = sy.transform_point(1.0, 1.0);
        assert!(close(x, 1.0) && close(y, 2.0), "({x}, {y})");
    }

    #[test]
    fn axes_and_scale_factor() {
        let t = Affine2D::scale(3.0, 4.0);
        assert_eq!(t.x_axis(), (3.0, 0.0));
        assert_eq!(t.y_axis(), (0.0, 4.0));
        assert!(close(t.scale_factor(), 12.0f32.powf(1.0 / 3.0)));
        // A rotation preserves lengths, so the scale factor is 1.
        assert!(close(Affine2D::rotate(1.0).scale_factor(), 1.0));
        // A reflection keeps the same size.
        assert!(close(
            Affine2D::scale(-3.0, 4.0).scale_factor(),
            t.scale_factor()
        ));
        // And the identity is unit scale.
        assert!(close(Affine2D::IDENTITY.scale_factor(), 1.0));
    }

    #[test]
    fn new_is_the_six_value_constructor() {
        let t = Affine2D::new(2.0, 3.0, 4.0, 5.0, 6.0, 7.0);
        assert_eq!(t.a, 2.0);
        assert_eq!(t.b, 3.0);
        assert_eq!(t.c, 4.0);
        assert_eq!(t.d, 5.0);
        assert_eq!(t.e, 6.0);
        assert_eq!(t.f, 7.0);
    }

    #[test]
    fn transform_ignores_translation_for_vectors() {
        let t = Affine2D::new(2.0, 0.0, 0.0, 3.0, 100.0, 200.0);
        assert_eq!(t.transform_vector(1.0, 1.0), (2.0, 3.0));
        assert_eq!(t.transform_point(1.0, 1.0), (102.0, 203.0));
    }

    #[test]
    fn euler_constants_are_finite() {
        // The rotate implementation must not silently produce NaN for any
        // finite input.
        for i in -8..8 {
            let t = Affine2D::rotate(i as f32 * FRAC_PI_4);
            assert!(t.a.is_finite() && t.b.is_finite(), "{i}");
            assert!(t.determinant().is_finite());
        }
        assert!((Affine2D::rotate(E).a - E.cos()).abs() < 1e-6);
    }
}
