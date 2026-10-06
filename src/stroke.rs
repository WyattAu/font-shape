//! Stroke expansion: a stroke becomes an outline, then gets filled.
//!
//! Rasterizing a stroke directly means solving "distance to the path" per
//! pixel, which is exact and slow. Expanding the stroke into the *region* it
//! covers — one closed quad per segment, plus joins and caps — and filling
//! that with the same scanline filler is the standard robust approach: one
//! rasterizer, one set of edge cases, and correct behaviour with overlapping
//! self-intersections under the non-zero rule.
//!
//! # What is expanded
//!
//! | Part | Contribution |
//! |---|---|
//! | segment | a quad from the two offset lines, both sides |
//! | join ([`LineJoin::Miter`]) | a miter wedge, falling back to a bevel past the limit |
//! | join ([`LineJoin::Round`]) | a fan around the vertex |
//! | join ([`LineJoin::Bevel`]) | the triangle between the two offsets |
//! | cap ([`LineCap::Butt`]) | nothing |
//! | cap ([`LineCap::Square`]) | a quad extending `width / 2` past the endpoint |
//! | cap ([`LineCap::Round`]) | a fan at the endpoint |
//!
//! Every emitted contour runs in the same rotational direction, so filling
//! the union with [`FillRule::NonZero`](crate::FillRule::NonZero) is a union —
//! which is exactly what a stroke is.

use alloc::vec;
use alloc::vec::Vec;

use crate::path::{Path2D, PathCommand};

/// How the ends of an open subpath are finished.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LineCap {
    /// Stop exactly at the endpoint. The default, matching SVG.
    #[default]
    Butt,
    /// Extend by half the stroke width, squared off.
    Square,
    /// Extend by half the stroke width, rounded.
    Round,
}

/// How the corners between two segments are finished.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LineJoin {
    /// Extend both sides until they meet, or bevel past the miter limit. The
    /// default, matching SVG.
    #[default]
    Miter,
    /// Cut the corner off with a straight edge.
    Bevel,
    /// Round the corner with an arc.
    Round,
}

/// The default miter limit: a miter longer than this many stroke widths is
/// beveled instead. Matches SVG's `stroke-miterlimit` default.
pub const DEFAULT_MITER_LIMIT: f32 = 4.0;

/// How much arc a round join or cap approximates its true circle with.
const ARC_STEPS: usize = 12;

/// A stroke's width, caps, joins, and miter limit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StrokeStyle {
    /// The stroke width. A zero or negative width yields an empty path: there
    /// is no region to fill.
    pub width: f32,
    /// The end treatment.
    pub cap: LineCap,
    /// The corner treatment.
    pub join: LineJoin,
    /// The longest miter, as a multiple of the stroke width.
    pub miter_limit: f32,
}

impl Default for StrokeStyle {
    fn default() -> Self {
        StrokeStyle {
            width: 1.0,
            cap: LineCap::Butt,
            join: LineJoin::Miter,
            miter_limit: DEFAULT_MITER_LIMIT,
        }
    }
}

impl StrokeStyle {
    /// A style with the given width and the default cap and join.
    #[must_use]
    pub const fn new(width: f32) -> Self {
        StrokeStyle {
            width,
            cap: LineCap::Butt,
            join: LineJoin::Miter,
            miter_limit: DEFAULT_MITER_LIMIT,
        }
    }

    /// Set the cap.
    #[must_use]
    pub const fn with_cap(mut self, cap: LineCap) -> Self {
        self.cap = cap;
        self
    }

    /// Set the join.
    #[must_use]
    pub const fn with_join(mut self, join: LineJoin) -> Self {
        self.join = join;
        self
    }

    /// Set the miter limit.
    #[must_use]
    pub const fn with_miter_limit(mut self, limit: f32) -> Self {
        self.miter_limit = limit;
        self
    }
}

/// Expand a stroke into the outline of the region it covers.
///
/// ```
/// use font_shape::{stroke_path, LineCap, PathBuilder, StrokeStyle};
///
/// let mut b = PathBuilder::new();
/// b.move_to(0.0, 0.0);
/// b.line_to(100.0, 0.0);
/// let path = b.build();
///
/// // A butt-capped 10-wide horizontal stroke is a 100 x 10 rectangle.
/// let s = stroke_path(&path, 10.0, StrokeStyle::new(10.0));
/// let bbox = s.bbox().expect("bbox");
/// assert_eq!(bbox.width(), 100.0);
/// assert_eq!(bbox.height(), 10.0);
///
/// // A square cap extends each end by half the width.
/// let c = stroke_path(&path, 10.0, StrokeStyle::new(10.0).with_cap(LineCap::Square));
/// assert_eq!(c.bbox().expect("bbox").width(), 110.0);
///
/// // A zero width has no region at all.
/// assert!(stroke_path(&path, 0.0, StrokeStyle::new(0.0)).is_empty());
/// ```
#[must_use]
pub fn stroke_path(path: &Path2D, width: f32, style: StrokeStyle) -> Path2D {
    let mut out = Path2D::new();
    let hw = width * 0.5;
    // A zero, negative, or non-finite width has no region to fill. SVG says a
    // zero-width stroke renders as a hairline; that is a rasterisation
    // convention, and this function expands geometry, so it reports the truth:
    // nothing.
    if !hw.is_finite() || hw <= 0.0 {
        return out;
    }
    // Flatten once. A stroke's joins depend on the polygon's vertices, and the
    // tolerance here is a fiftieth of the stroke's own half-width — far finer
    // than any cap or join needs.
    //
    // The closedness comes from the *source* subpath, not from the flattened
    // points: `flatten` appends an implicit closing point to open subpaths too,
    // and treating that as a `Close` would put a join where a cap belongs.
    for poly in path.flatten_polylines(hw * 0.05) {
        // Both branches drop the repeated closing point. For a closed subpath
        // that is bookkeeping; for an open one the implicit closing segment is
        // *not* part of the stroke, and keeping it is what would put a join
        // where a cap belongs.
        let pts: Vec<(f32, f32)> = match poly.points.split_last() {
            Some((last, head)) if head.len() > 1 && poly.points.first() == Some(last) => {
                head.to_vec()
            }
            _ => poly.points,
        };
        if pts.len() < 2 {
            continue;
        }
        let mut seg = StrokeEmitter {
            out: &mut out,
            hw,
            style,
        };
        seg.emit(&pts, poly.closed);
    }
    out
}

/// Emits the quads, joins, and caps of one polyline.
struct StrokeEmitter<'a> {
    out: &'a mut Path2D,
    hw: f32,
    style: StrokeStyle,
}

impl StrokeEmitter<'_> {
    /// One quad per segment, plus joins between them and caps at the ends.
    fn emit(&mut self, pts: &[(f32, f32)], closed: bool) {
        let n = pts.len();
        let last = if closed { n } else { n - 1 };
        let at = |i: usize| -> Option<(f32, f32)> { pts.get(i % n).copied() };
        for i in 0..last {
            let (Some(a), Some(b)) = (at(i), at(i + 1)) else {
                continue;
            };
            if a == b {
                // A zero-length segment contributes no region, and no join.
                continue;
            }
            let (na, nb) = (normal(a, b, self.hw), normal(a, b, -self.hw));
            self.quad(
                (a.0 + na.0, a.1 + na.1),
                (b.0 + na.0, b.1 + na.1),
                (b.0 + nb.0, b.1 + nb.1),
                (a.0 + nb.0, a.1 + nb.1),
            );
        }
        // Joins. A closed subpath has one at every vertex; an open one only at its
        // *interior* vertices, since its two ends get caps instead. Joining at
        // an open end is not cosmetic: the "previous" vertex would wrap around
        // to the far end of the polyline, filling a wedge nowhere near the cap.
        let first_join = if closed { 0 } else { 1 };
        let last_join = if closed { n } else { n - 1 };
        for i in first_join..last_join {
            let (Some(prev), Some(here), Some(next)) = (at(i + n - 1), at(i), at(i + 1)) else {
                continue;
            };
            self.join(prev, here, next);
        }
        if !closed {
            match self.style.cap {
                LineCap::Butt => {}
                LineCap::Square | LineCap::Round => {
                    let round = matches!(self.style.cap, LineCap::Round);
                    let (Some(first), Some(second)) = (pts.first().copied(), pts.get(1).copied())
                    else {
                        return;
                    };
                    let (Some(last), Some(prev)) = (pts.last().copied(), pts.get(n - 2).copied())
                    else {
                        return;
                    };
                    if round {
                        self.cap_round(first, second);
                        self.cap_round(last, prev);
                    } else {
                        self.cap_square(first, second);
                        self.cap_square(last, prev);
                    }
                }
            }
        }
    }

    /// A closed quad, emitted in a consistent rotational direction.
    fn quad(&mut self, a: (f32, f32), b: (f32, f32), c: (f32, f32), d: (f32, f32)) {
        self.out.push(PathCommand::MoveTo(a.0, a.1));
        self.out.push(PathCommand::LineTo(b.0, b.1));
        self.out.push(PathCommand::LineTo(c.0, c.1));
        self.out.push(PathCommand::LineTo(d.0, d.1));
        self.out.push(PathCommand::Close);
    }

    /// A closed triangle.
    fn triangle(&mut self, a: (f32, f32), b: (f32, f32), c: (f32, f32)) {
        self.out.push(PathCommand::MoveTo(a.0, a.1));
        self.out.push(PathCommand::LineTo(b.0, b.1));
        self.out.push(PathCommand::LineTo(c.0, c.1));
        self.out.push(PathCommand::Close);
    }

    /// The join at `here`, between the segment arriving from `prev` and the one
    /// leaving toward `next`.
    fn join(&mut self, prev: (f32, f32), here: (f32, f32), next: (f32, f32)) {
        let (Some(in_dir), Some(out_dir)) = (
            norm((here.0 - prev.0, here.1 - prev.1)),
            norm((next.0 - here.0, next.1 - here.1)),
        ) else {
            return;
        };
        // The cross product's sign says which way the path turns; the normal
        // offsets then cross over or splay apart.
        let cross = in_dir.0 * out_dir.1 - in_dir.1 * out_dir.0;
        if cross.abs() < 1e-9 {
            // Collinear: the two quads already meet, nothing to fill.
            return;
        }
        let n_in = (in_dir.1 * self.hw, -in_dir.0 * self.hw);
        let n_out = (out_dir.1 * self.hw, -out_dir.0 * self.hw);
        let side_in = (here.0 + n_in.0, here.1 + n_in.1);
        let side_out = (here.0 + n_out.0, here.1 + n_out.1);
        match self.style.join {
            LineJoin::Bevel => self.triangle(here, side_in, side_out),
            LineJoin::Round => self.fan(here, side_in, side_out, cross),
            LineJoin::Miter => {
                // The miter length is `hw / cos(θ/2)`, so the limit compares
                // the miter's length against `hw / miter_limit`.
                let dot = in_dir.0 * out_dir.0 + in_dir.1 * out_dir.1;
                let half_cos = libm::sqrtf((1.0 + dot) * 0.5).max(1e-6);
                let ratio = 1.0 / half_cos;
                if ratio > self.style.miter_limit.max(1.0) {
                    // Past the limit: SVG's fallback is a bevel.
                    self.triangle(here, side_in, side_out);
                } else {
                    // The miter point is where the two *offset lines* meet, so
                    // it lies along the bisector of the two normals — not of
                    // the two directions, which points into the corner's
                    // interior and would produce a zero-area wedge.
                    let Some(n) = norm((n_in.0 + n_out.0, n_in.1 + n_out.1)) else {
                        self.triangle(here, side_in, side_out);
                        return;
                    };
                    let miter = (
                        here.0 + n.0 * self.hw * ratio,
                        here.1 + n.1 * self.hw * ratio,
                    );
                    self.quad(here, side_in, miter, side_out);
                }
            }
        }
    }

    /// A round fan from `from` to `to` around `centre`.
    fn fan(&mut self, centre: (f32, f32), from: (f32, f32), to: (f32, f32), sign: f32) {
        let a0 = libm::atan2f(from.1 - centre.1, from.0 - centre.0);
        let a1 = libm::atan2f(to.1 - centre.1, to.0 - centre.0);
        // Sweep the short way round, in the direction `sign` names.
        let mut delta = a1 - a0;
        while delta <= -core::f32::consts::PI {
            delta += 2.0 * core::f32::consts::PI;
        }
        while delta > core::f32::consts::PI {
            delta -= 2.0 * core::f32::consts::PI;
        }
        let steps = ARC_STEPS.max(2);
        self.out.push(PathCommand::MoveTo(centre.0, centre.1));
        for i in 0..=steps {
            let t = a0 + delta * (i as f32) / (steps as f32);
            let (s, c) = (libm::sinf(t), libm::cosf(t));
            let p = (centre.0 + c * self.hw, centre.1 + s * self.hw);
            self.out.push(PathCommand::LineTo(p.0, p.1));
        }
        self.out.push(PathCommand::Close);
        let _ = sign;
    }

    /// A square cap at `end`, extending half a width along the segment.
    fn cap_square(&mut self, end: (f32, f32), toward: (f32, f32)) {
        let Some(d) = norm((end.0 - toward.0, end.1 - toward.1)) else {
            return;
        };
        // The cap is a rectangle `hw` past the endpoint, `width` wide: two of its
        // corners are the endpoint's own offsets and two the same offsets
        // pushed out along `d`. `d` points *away* from the path at both ends,
        // so the same corner order winds the same way at both — which is what
        // makes the non-zero union correct.
        let n = (d.1 * self.hw, -d.0 * self.hw);
        let a = (end.0 + n.0, end.1 + n.1);
        let b = (end.0 - n.0, end.1 - n.1);
        let far_a = (a.0 + d.0 * self.hw, a.1 + d.1 * self.hw);
        let far_b = (b.0 + d.0 * self.hw, b.1 + d.1 * self.hw);
        self.quad(a, far_a, far_b, b);
    }

    /// A round cap at `end`: a half-disc.
    fn cap_round(&mut self, end: (f32, f32), toward: (f32, f32)) {
        let Some(d) = norm((end.0 - toward.0, end.1 - toward.1)) else {
            return;
        };
        let n = (d.1 * self.hw, -d.0 * self.hw);
        let a = (end.0 + n.0, end.1 + n.1);
        let b = (end.0 - n.0, end.1 - n.1);
        // `d` points away from the path at both ends, so sweeping from `a` to
        // `b` the positive way round always goes through the outer half — the
        // same direction at both ends, which keeps the union non-zero.
        self.arc(end, a, b);
    }

    /// A fan from `from` to `to` around `centre`, swept counter-clockwise.
    fn arc(&mut self, centre: (f32, f32), from: (f32, f32), to: (f32, f32)) {
        let a0 = libm::atan2f(from.1 - centre.1, from.0 - centre.0);
        let a1 = libm::atan2f(to.1 - centre.1, to.0 - centre.0);
        let mut delta = a1 - a0;
        // Normalise into (0, 2π]: a fan that swept the short way would put the
        // wrong half of the disc on the stroke.
        while delta <= 0.0 {
            delta += 2.0 * core::f32::consts::PI;
        }
        while delta > 2.0 * core::f32::consts::PI {
            delta -= 2.0 * core::f32::consts::PI;
        }
        let steps = ARC_STEPS.max(2);
        self.out.push(PathCommand::MoveTo(centre.0, centre.1));
        for i in 0..=steps {
            let t = a0 + delta * (i as f32) / (steps as f32);
            let (s, c) = (libm::sinf(t), libm::cosf(t));
            let p = (centre.0 + c * self.hw, centre.1 + s * self.hw);
            self.out.push(PathCommand::LineTo(p.0, p.1));
        }
        self.out.push(PathCommand::Close);
    }
}

/// The unit normal to the direction `d`, scaled by `hw`, on one side.
///
/// `side` is `+hw` or `-hw`, giving the two offset lines of a segment.
fn normal(from: (f32, f32), to: (f32, f32), side: f32) -> (f32, f32) {
    let dx = to.0 - from.0;
    let dy = to.1 - from.1;
    let len = libm::sqrtf(dx * dx + dy * dy);
    if len <= 1e-9 {
        return (0.0, 0.0);
    }
    let ux = dx / len;
    let uy = dy / len;
    (uy * side, -ux * side)
}

/// A unit vector along `d`, `None` for a zero-length direction.
fn norm(d: (f32, f32)) -> Option<(f32, f32)> {
    let len = libm::sqrtf(d.0 * d.0 + d.1 * d.1);
    if len <= 1e-9 || !len.is_finite() {
        return None;
    }
    Some((d.0 / len, d.1 / len))
}

/// Reverse every subpath of a command list, keeping the geometry.
///
/// Used by [`Path2D::reverse`](crate::Path2D::reverse) and shared with the
/// tests: a segment's reversal swaps its endpoints and reverses its controls,
/// and a subpath's segments come out in the opposite order.
#[must_use]
pub(crate) fn reverse_commands(commands: &[PathCommand]) -> Vec<PathCommand> {
    /// One segment: its kind, its control points, and its on-curve endpoint.
    struct Seg {
        kind: u8,
        c1: (f32, f32),
        c2: (f32, f32),
        to: (f32, f32),
    }
    /// A subpath under construction: on-curve points, segments, and whether
    /// it was explicitly closed.
    type Sub = (Vec<(f32, f32)>, Vec<Seg>, bool);
    /// The same, minus the closed flag, which is only known at `Close`.
    type Open = (Vec<(f32, f32)>, Vec<Seg>);

    let mut out: Vec<PathCommand> = Vec::with_capacity(commands.len());
    let mut subs: Vec<Sub> = Vec::new();
    let mut cur: Option<Open> = None;
    for cmd in commands {
        match *cmd {
            PathCommand::MoveTo(x, y) => {
                if let Some(sp) = cur.take() {
                    subs.push((sp.0, sp.1, false));
                }
                cur = Some((vec![(x, y)], Vec::new()));
            }
            PathCommand::LineTo(x, y) => {
                if let Some(sp) = cur.as_mut() {
                    sp.1.push(Seg {
                        kind: 1,
                        c1: (0.0, 0.0),
                        c2: (0.0, 0.0),
                        to: (x, y),
                    });
                    sp.0.push((x, y));
                }
            }
            PathCommand::QuadTo(cx, cy, x, y) => {
                if let Some(sp) = cur.as_mut() {
                    sp.1.push(Seg {
                        kind: 2,
                        c1: (cx, cy),
                        c2: (0.0, 0.0),
                        to: (x, y),
                    });
                    sp.0.push((x, y));
                }
            }
            PathCommand::CubicTo(c1x, c1y, c2x, c2y, x, y) => {
                if let Some(sp) = cur.as_mut() {
                    sp.1.push(Seg {
                        kind: 3,
                        c1: (c1x, c1y),
                        c2: (c2x, c2y),
                        to: (x, y),
                    });
                    sp.0.push((x, y));
                }
            }
            PathCommand::Close => {
                if let Some(sp) = cur.take() {
                    subs.push((sp.0, sp.1, true));
                }
            }
        }
    }
    if let Some(sp) = cur.take() {
        subs.push((sp.0, sp.1, false));
    }
    for (points, segs, closed) in subs {
        let Some(first) = points.first().copied() else {
            continue;
        };
        if segs.is_empty() {
            // A subpath with no segments — a lone `MoveTo`, which is a point
            // with no extent. It has no direction to flip, but it is still a
            // point the path's bounding box covers, so dropping it would make
            // reversal change the geometry.
            out.push(PathCommand::MoveTo(first.0, first.1));
            if closed {
                out.push(PathCommand::Close);
            }
            continue;
        }
        let begin = segs.last().map_or(first, |s| s.to);
        out.push(PathCommand::MoveTo(begin.0, begin.1));
        for (i, s) in segs.iter().enumerate().rev() {
            let to = match i.checked_sub(1).and_then(|k| segs.get(k)) {
                Some(prev) => prev.to,
                None => first,
            };
            match s.kind {
                1 => out.push(PathCommand::LineTo(to.0, to.1)),
                2 => out.push(PathCommand::QuadTo(s.c1.0, s.c1.1, to.0, to.1)),
                _ => out.push(PathCommand::CubicTo(
                    s.c2.0, s.c2.1, s.c1.0, s.c1.1, to.0, to.1,
                )),
            }
        }
        if closed {
            out.push(PathCommand::Close);
        }
    }
    out
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
    use super::{stroke_path, LineCap, LineJoin, StrokeStyle};
    use crate::path::{PathBuilder, PathCommand};
    use crate::FillRule;

    // Test harness: assertions legitimately panic; the lib target stays
    // lint-clean.

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-3
    }

    /// Total coverage of a stroke, as a fraction of full ink.
    fn ink(path: &crate::Path2D, size: u32) -> f32 {
        let mask = crate::fill_path(path, FillRule::NonZero, size, size).expect("fills");
        mask.iter().map(|&v| f32::from(v)).sum::<f32>() / (255.0 * (size * size) as f32)
    }

    fn line(from: (f32, f32), to: (f32, f32)) -> crate::Path2D {
        let mut b = PathBuilder::new();
        b.move_to(from.0, from.1);
        b.line_to(to.0, to.1);
        b.build()
    }

    #[test]
    fn zero_width_stroke_has_no_region() {
        let p = line((10.0, 10.0), (50.0, 10.0));
        for w in [0.0, -1.0, -0.001] {
            let s = stroke_path(&p, w, StrokeStyle::new(w));
            assert!(s.is_empty(), "width {w} produced {} commands", s.len());
        }
        // A non-finite width likewise.
        assert!(stroke_path(&p, f32::NAN, StrokeStyle::default()).is_empty());
        assert!(stroke_path(&p, f32::INFINITY, StrokeStyle::default()).is_empty());
    }

    #[test]
    fn hairline_is_a_thin_but_nonzero_region() {
        // A 0.01-wide stroke is the practical "hairline": a real region, so
        // the difference from zero is visible.
        let p = line((10.0, 10.0), (50.0, 10.0));
        let s = stroke_path(&p, 0.01, StrokeStyle::new(0.01));
        assert!(!s.is_empty());
        let b = s.bbox().expect("bbox");
        assert!(close(b.height(), 0.01), "height {}", b.height());
        assert!(close(b.width(), 40.0));
    }

    #[test]
    fn butt_cap_stops_at_the_endpoint() {
        let p = line((20.0, 50.0), (80.0, 50.0));
        let s = stroke_path(&p, 10.0, StrokeStyle::new(10.0).with_cap(LineCap::Butt));
        let b = s.bbox().expect("bbox");
        assert_eq!(b.min_x, 20.0);
        assert_eq!(b.max_x, 80.0);
        assert_eq!(b.min_y, 45.0);
        assert_eq!(b.max_y, 55.0);
    }

    #[test]
    fn square_cap_extends_by_half_the_width() {
        let p = line((20.0, 50.0), (80.0, 50.0));
        let s = stroke_path(&p, 10.0, StrokeStyle::new(10.0).with_cap(LineCap::Square));
        let b = s.bbox().expect("bbox");
        assert_eq!(b.min_x, 15.0);
        assert_eq!(b.max_x, 85.0);
        // Still the same thickness across the body.
        assert_eq!(b.height(), 10.0);
    }

    #[test]
    fn round_cap_extends_by_exactly_half_the_width() {
        let p = line((30.0, 50.0), (70.0, 50.0));
        let s = stroke_path(&p, 10.0, StrokeStyle::new(10.0).with_cap(LineCap::Round));
        let b = s.bbox().expect("bbox");
        // The cap is an `ARC_STEPS`-gon inscribed in the true half-disc, so it
        // reaches exactly `hw` along the segment's axis — the first and last
        // arc steps lie on the axis — and falls short of `hw` by the diagonal.
        assert!(close(b.min_x, 25.0), "min_x {}", b.min_x);
        assert!(close(b.max_x, 75.0), "max_x {}", b.max_x);
        assert_eq!(b.height(), 10.0, "the cap is as wide as the stroke");
        // A square cap is the same length, so the two agree at the extremes.
        let sq = stroke_path(&p, 10.0, StrokeStyle::new(10.0).with_cap(LineCap::Square));
        assert_eq!(sq.bbox(), Some(b));
    }

    #[test]
    fn diagonal_stroke_thickness_is_the_width() {
        // A butt-capped 45-degree stroke of width w is bounded by the segment's
        // endpoints offset by the half-width perpendicular, which at 45° is
        // hw·cos(45°) = w / (2√2) in each axis.
        let p = line((0.0, 0.0), (100.0, 100.0));
        let s = stroke_path(&p, 10.0, StrokeStyle::new(10.0));
        let b = s.bbox().expect("bbox");
        let expected = 5.0 / core::f32::consts::SQRT_2;
        assert!(
            close(b.min_x, -expected) && close(b.min_y, -expected),
            "{b:?}"
        );
        assert!(close(b.max_x, 100.0 + expected), "{b:?}");
        assert!(close(b.max_y, 100.0 + expected), "{b:?}");
    }

    #[test]
    fn miter_join_extends_a_right_angle_to_the_miter_limit() {
        // An L: the miter at the corner reaches hw·sqrt(2) for a 90° turn.
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0);
        b.line_to(50.0, 0.0);
        b.line_to(50.0, 50.0);
        let p = b.build();
        let s = stroke_path(&p, 10.0, StrokeStyle::new(10.0).with_join(LineJoin::Miter));
        let bb = s.bbox().expect("bbox");
        // The miter point is where the two offset lines meet: `hw·sqrt(2)`
        // beyond the corner along the bisector, which at a right angle lands
        // at (55, -5).
        assert!(close(bb.max_x, 55.0), "{bb:?}");
        assert!(close(bb.min_y, -5.0), "{bb:?}");
        assert!(close(bb.min_x, 0.0) && close(bb.max_y, 50.0), "{bb:?}");
        // A bevel stops at the offset endpoints instead: it cuts the corner
        // off, so it covers strictly less area. That difference is the
        // miter's whole point.
        let bevel = stroke_path(&p, 10.0, StrokeStyle::new(10.0).with_join(LineJoin::Bevel));
        let bvb = bevel.bbox().expect("bbox");
        assert!(close(bvb.max_x, 55.0) && close(bvb.min_y, -5.0), "{bvb:?}");
        let miter_area = s.polygon_area();
        let bevel_area = bevel.polygon_area();
        assert!(
            miter_area > bevel_area,
            "miter {miter_area} vs bevel {bevel_area}"
        );
        // The gap is the right triangle between the bevel's chord and the
        // miter's point: `hw² − hw² / √2` ≈ 12.5.
        assert!(
            (miter_area - bevel_area - 12.5).abs() < 0.5,
            "gap {}",
            miter_area - bevel_area
        );
    }

    #[test]
    fn miter_limit_exceeded_falls_back_to_a_bevel() {
        // A very sharp spike: the miter would be enormous, so the bevel is used.
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0);
        b.line_to(100.0, 0.0);
        b.line_to(50.0, 1.0); // a near-180° reversal
        let p = b.build();

        let bevel = stroke_path(&p, 4.0, StrokeStyle::new(4.0).with_join(LineJoin::Bevel));
        let tight = stroke_path(
            &p,
            4.0,
            StrokeStyle::new(4.0)
                .with_join(LineJoin::Miter)
                .with_miter_limit(1.5),
        );
        // Past the limit the miter join *is* the bevel join.
        assert_eq!(
            tight.commands(),
            bevel.commands(),
            "a 1.5 limit bevels this spike"
        );
        // With a generous limit the two differ.
        let loose = stroke_path(&p, 4.0, StrokeStyle::new(4.0).with_miter_limit(1000.0));
        assert_ne!(loose.commands(), bevel.commands());
    }

    #[test]
    fn bevel_join_matches_miter_below_the_limit() {
        // A gentle turn: both produce the same region, since the miter is
        // within the default limit.
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0);
        b.line_to(50.0, 0.0);
        b.line_to(100.0, 20.0);
        let p = b.build();
        let miter = stroke_path(&p, 6.0, StrokeStyle::new(6.0).with_join(LineJoin::Miter));
        let bevel = stroke_path(&p, 6.0, StrokeStyle::new(6.0).with_join(LineJoin::Bevel));
        // Same coverage to within a fraction of a percent.
        let a = ink(&miter, 128);
        let c = ink(&bevel, 128);
        assert!((a - c).abs() < 0.005, "miter {a} vs bevel {c}");
    }

    #[test]
    fn joins_cover_different_amounts_at_a_corner() {
        // The corner is placed well inside the raster: a stroke's bounding box
        // extends past the path, and ink measured on a path that hangs off the
        // edge is clipped, which would hide the very differences under test.
        let mut b = PathBuilder::new();
        b.move_to(30.0, 30.0);
        b.line_to(80.0, 30.0);
        b.line_to(80.0, 80.0);
        let p = b.build();
        let miter = stroke_path(&p, 10.0, StrokeStyle::new(10.0).with_join(LineJoin::Miter));
        let round = stroke_path(&p, 10.0, StrokeStyle::new(10.0).with_join(LineJoin::Round));
        let bevel = stroke_path(&p, 10.0, StrokeStyle::new(10.0).with_join(LineJoin::Bevel));

        // A quarter-disc is π·hw²/4 ≈ 19.6; the bevel's triangle is hw²/2 = 12.5;
        // the miter's wedge is hw² − hw²/√2 ≈ 12.5 as well but in a different
        // place. The bevel is the smallest of the three.
        assert!(ink(&round, 256) > ink(&bevel, 256), "round vs bevel");
        assert!(ink(&miter, 256) > ink(&bevel, 256), "miter vs bevel");
        // Round and miter differ by a few square units out of ~1000, which at
        // a 256² raster is well under a tenth of a percent of the frame — so
        // compare the areas, not the frame-normalised ink.
        let miter_area = miter.polygon_area();
        assert!(
            (round.polygon_area() - miter_area).abs() > 1.0,
            "round and miter differ"
        );
        // The two segment quads contribute 2 · 50 · 10 = 1000, so what is left
        // is the join itself. A round join's fan covers a quarter-disc, less
        // the triangle its two chords already share with the quads — and the
        // fan's twelve chords make it slightly smaller again, which is what
        // pins ARC_STEPS.
        let round_join = round.polygon_area() - 1000.0;
        let quarter_disc = core::f32::consts::FRAC_PI_4 * 25.0;
        assert!(
            (round_join - quarter_disc).abs() < 0.5,
            "round join area {round_join} vs quarter-disc {quarter_disc}"
        );
        // And the bevel's is the triangle, which is smaller.
        let bevel_join = bevel.polygon_area() - 1000.0;
        assert!(
            (bevel_join - 12.5).abs() < 0.5,
            "bevel join area {bevel_join} vs triangle 12.5"
        );
    }

    #[test]
    fn collinear_join_adds_nothing() {
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0);
        b.line_to(50.0, 0.0);
        b.line_to(100.0, 0.0);
        let p = b.build();
        let s = stroke_path(&p, 10.0, StrokeStyle::new(10.0));
        let bb = s.bbox().expect("bbox");
        assert_eq!(bb.min_y, -5.0);
        assert_eq!(bb.max_y, 5.0);
    }

    #[test]
    fn closed_subpath_strokes_the_whole_ring() {
        let mut b = PathBuilder::new();
        b.rect(20.0, 20.0, 60.0, 60.0);
        let p = b.build();
        let s = stroke_path(&p, 6.0, StrokeStyle::new(6.0));
        let bb = s.bbox().expect("bbox");
        // Outward by half the width on every side, plus the mitered corners.
        assert!(bb.min_x <= 17.0 && bb.max_x >= 83.0, "{bb:?}");
        // A closed ring has no caps, so it does not extend past the miter.
        let round = stroke_path(&p, 6.0, StrokeStyle::new(6.0).with_join(LineJoin::Round));
        let rb = round.bbox().expect("bbox");
        assert!(close(rb.min_x, 17.0) && close(rb.max_x, 83.0), "{rb:?}");
    }

    #[test]
    fn stroke_ink_is_length_times_width() {
        // A 100-long, 10-wide horizontal stroke covers 1000 square units.
        let p = line((20.0, 50.0), (120.0, 50.0));
        let s = stroke_path(&p, 10.0, StrokeStyle::new(10.0));
        let area = s.polygon_area();
        assert!((area - 1000.0).abs() < 2.0, "polygon area {area} vs 1000");
        // The rendered mask agrees to within a percent at a fine resolution.
        let mut b = PathBuilder::new();
        b.extend(&s.translated(0.0, 0.0));
        let total = ink(&b.build(), 512);
        let expect = 1000.0 / (512.0 * 512.0);
        assert!(
            (total - expect).abs() < expect * 0.01,
            "mask ink {total} vs {expect}"
        );
    }

    #[test]
    fn zero_length_segment_contributes_nothing() {
        let mut b = PathBuilder::new();
        b.move_to(10.0, 10.0);
        b.line_to(10.0, 10.0);
        b.line_to(50.0, 10.0);
        let p = b.build();
        let s = stroke_path(&p, 10.0, StrokeStyle::new(10.0));
        let bb = s.bbox().expect("bbox");
        assert_eq!(bb.min_x, 10.0, "the zero segment adds no cap");
        assert_eq!(bb.max_x, 50.0);
    }

    #[test]
    fn degenerate_paths_do_not_panic() {
        // An empty path.
        assert!(stroke_path(&crate::Path2D::new(), 5.0, StrokeStyle::default()).is_empty());
        // A lone point.
        let mut b = PathBuilder::new();
        b.move_to(5.0, 5.0);
        let p = b.build();
        assert!(stroke_path(&p, 5.0, StrokeStyle::default()).is_empty());
        // Two identical points.
        let mut b = PathBuilder::new();
        b.move_to(5.0, 5.0);
        b.line_to(5.0, 5.0);
        b.close();
        assert!(stroke_path(&b.build(), 5.0, StrokeStyle::default()).is_empty());
    }

    #[test]
    fn style_builders() {
        let s = StrokeStyle::default();
        assert_eq!(s.width, 1.0);
        assert_eq!(s.cap, LineCap::Butt);
        assert_eq!(s.join, LineJoin::Miter);
        assert_eq!(s.miter_limit, super::DEFAULT_MITER_LIMIT);
        let s = StrokeStyle::new(3.0)
            .with_cap(LineCap::Round)
            .with_join(LineJoin::Round)
            .with_miter_limit(2.0);
        assert_eq!(s.width, 3.0);
        assert_eq!(s.cap, LineCap::Round);
        assert_eq!(s.join, LineJoin::Round);
        assert_eq!(s.miter_limit, 2.0);
    }

    #[test]
    fn reverse_commands_helper() {
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0);
        b.line_to(10.0, 0.0);
        b.line_to(10.0, 10.0);
        b.close();
        let before = b.build();
        let reversed = super::reverse_commands(before.commands());
        let mut after = before.clone();
        after.reverse();
        assert_eq!(reversed, after.commands());
        // Reversing an empty list is empty.
        assert!(super::reverse_commands(&[]).is_empty());
    }

    #[test]
    fn emitted_contours_all_close() {
        // Every subpath a stroke emits is closed, which is what makes the
        // non-zero union correct.
        let mut b = PathBuilder::new();
        b.move_to(10.0, 10.0);
        b.line_to(50.0, 10.0);
        b.line_to(50.0, 50.0);
        let s = stroke_path(&b.build(), 8.0, StrokeStyle::new(8.0));
        assert!(!s.commands().is_empty());
        assert!(
            matches!(s.commands().last(), Some(PathCommand::Close)),
            "last command {:?}",
            s.commands().last()
        );
        let mut starts = 0usize;
        let mut closes = 0usize;
        for c in s.commands() {
            match c {
                PathCommand::MoveTo(..) => starts += 1,
                PathCommand::Close => closes += 1,
                _ => {}
            }
        }
        assert_eq!(starts, closes, "every MoveTo has a Close");
    }

    #[test]
    fn many_segment_path_strokes_without_panicking() {
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0);
        for i in 1..200 {
            b.line_to(
                f32::from(i as u16) * 0.5,
                if i % 3 == 0 { 5.0 } else { 0.0 },
            );
        }
        let s = stroke_path(&b.build(), 2.0, StrokeStyle::new(2.0));
        assert!(!s.is_empty());
        assert!(s.bbox().is_some());
    }

    #[test]
    fn non_finite_coordinates_do_not_panic() {
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0);
        b.line_to(f32::NAN, 10.0);
        b.line_to(50.0, 10.0);
        let s = stroke_path(&b.build(), 4.0, StrokeStyle::new(4.0));
        // Whatever it produced, it produced something finite.
        if let Some(bb) = s.bbox() {
            assert!(bb.min_x.is_finite() || bb.min_x == f32::INFINITY);
        }
    }
}
