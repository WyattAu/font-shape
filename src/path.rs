//! Paths with an affine transform stack.
//!
//! A [`Path2D`] is a device-space sequence of subpaths: absolute line, quad,
//! and cubic segments. Curves are kept as curves — flattening happens at
//! rasterisation time, where the pixel grid makes the right tolerance known —
//! so a stroked-then-scaled path keeps its exact geometry.
//!
//! [`PathBuilder`] accumulates into a [`Path2D`] through a **transform stack**:
//! [`push_transform`](PathBuilder::push_transform) composes onto the current
//! transform, and [`pop_transform`](PathBuilder::pop_transform) restores the
//! one beneath it. An unbalanced pop is an error rather than a silent identity,
//! because dropping a transform silently would render the wrong glyph.
//!
//! ```
//! use font_shape::{Affine2D, PathBuilder};
//!
//! let mut b = PathBuilder::new();
//! b.push_transform(Affine2D::translate(10.0, 0.0));
//! b.rect(0.0, 0.0, 20.0, 20.0);
//! b.push_transform(Affine2D::scale(2.0, 2.0));
//! b.rect(0.0, 0.0, 5.0, 5.0);          // now at (10, 0), 10x10
//! b.pop_transform().expect("balanced");
//! b.pop_transform().expect("balanced");
//! assert!(b.pop_transform().is_err(), "an unbalanced pop is refused");
//!
//! // The rectangle drawn under the scale is twice the size, and offset by the
//! // translation.
//! let path = b.build();
//! assert_eq!(path.bbox().expect("bbox").max_x, 30.0);
//! ```

use alloc::vec;
use alloc::vec::Vec;

use crate::affine::Affine2D;
use crate::error::ShapeError;
use crate::stroke::{LineCap, LineJoin, StrokeStyle};

/// One resolved path segment, absolute.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PathCommand {
    /// Start a new subpath at this point.
    MoveTo(f32, f32),
    /// Straight segment to this point.
    LineTo(f32, f32),
    /// Quadratic segment: control point, then the on-curve endpoint.
    QuadTo(f32, f32, f32, f32),
    /// Cubic segment: two control points, then the endpoint.
    CubicTo(f32, f32, f32, f32, f32, f32),
    /// Close the current subpath.
    Close,
}

/// A device-space path: a list of subpaths of absolute segments.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Path2D {
    commands: Vec<PathCommand>,
}

impl Path2D {
    /// An empty path.
    #[must_use]
    pub fn new() -> Self {
        Path2D {
            commands: Vec::new(),
        }
    }

    /// The commands, in order.
    #[must_use]
    pub fn commands(&self) -> &[PathCommand] {
        &self.commands
    }

    /// Number of commands.
    #[must_use]
    pub fn len(&self) -> usize {
        self.commands.len()
    }

    /// True when the path holds no commands.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    /// Append a command verbatim.
    pub fn push(&mut self, cmd: PathCommand) -> &mut Self {
        self.commands.push(cmd);
        self
    }

    /// The exact bounding box, `None` for an empty path.
    ///
    /// Curves contribute their true extrema, the same way `font-model`'s
    /// outline bounding boxes do — a rasteriser that trusted a control hull
    /// would clip antialiased edges.
    #[must_use]
    pub fn bbox(&self) -> Option<Rect> {
        let mut b = Rect::empty();
        let mut cur = (0.0f32, 0.0f32);
        let mut any = false;
        for cmd in &self.commands {
            match *cmd {
                PathCommand::MoveTo(x, y) => {
                    b.extend(x, y);
                    cur = (x, y);
                    any = true;
                }
                PathCommand::LineTo(x, y) => {
                    b.extend(x, y);
                    cur = (x, y);
                    any = true;
                }
                PathCommand::QuadTo(cx, cy, x, y) => {
                    b.extend(x, y);
                    let p = [cur, (cx, cy), (x, y)];
                    for t in quad_roots(p[0].0, p[1].0, p[2].0) {
                        b.extend(quad_x(p, t), quad_y(p, t));
                    }
                    for t in quad_roots(p[0].1, p[1].1, p[2].1) {
                        b.extend(quad_x(p, t), quad_y(p, t));
                    }
                    cur = (x, y);
                    any = true;
                }
                PathCommand::CubicTo(c1x, c1y, c2x, c2y, x, y) => {
                    b.extend(x, y);
                    let p = [cur, (c1x, c1y), (c2x, c2y), (x, y)];
                    for t in cubic_roots(p[0].0, p[1].0, p[2].0, p[3].0) {
                        b.extend(cubic_x(p, t), cubic_y(p, t));
                    }
                    for t in cubic_roots(p[0].1, p[1].1, p[2].1, p[3].1) {
                        b.extend(cubic_x(p, t), cubic_y(p, t));
                    }
                    cur = (x, y);
                    any = true;
                }
                PathCommand::Close => {
                    cur = (0.0, 0.0);
                    any = true;
                }
            }
        }
        if any {
            Some(b)
        } else {
            None
        }
    }

    /// Apply an affine transform to every point, curves included.
    ///
    /// The result is a new path; the original is untouched. Control points
    /// transform exactly like endpoints, which is what makes a transformed
    /// curve the same curve as the untransformed one.
    #[must_use]
    pub fn transformed(&self, m: &Affine2D) -> Path2D {
        let mut out = Path2D::new();
        out.commands.reserve(self.commands.len());
        for cmd in &self.commands {
            let t = match *cmd {
                PathCommand::MoveTo(x, y) => {
                    let (x, y) = m.transform_point(x, y);
                    PathCommand::MoveTo(x, y)
                }
                PathCommand::LineTo(x, y) => {
                    let (x, y) = m.transform_point(x, y);
                    PathCommand::LineTo(x, y)
                }
                PathCommand::QuadTo(cx, cy, x, y) => {
                    let (cx, cy) = m.transform_point(cx, cy);
                    let (x, y) = m.transform_point(x, y);
                    PathCommand::QuadTo(cx, cy, x, y)
                }
                PathCommand::CubicTo(c1x, c1y, c2x, c2y, x, y) => {
                    let (c1x, c1y) = m.transform_point(c1x, c1y);
                    let (c2x, c2y) = m.transform_point(c2x, c2y);
                    let (x, y) = m.transform_point(x, y);
                    PathCommand::CubicTo(c1x, c1y, c2x, c2y, x, y)
                }
                PathCommand::Close => PathCommand::Close,
            };
            out.push(t);
        }
        out
    }

    /// Translate the whole path.
    #[must_use]
    pub fn translated(&self, dx: f32, dy: f32) -> Path2D {
        self.transformed(&Affine2D::translate(dx, dy))
    }

    /// Reverse every subpath's direction, keeping the geometry.
    pub fn reverse(&mut self) {
        let reversed = crate::stroke::reverse_commands(&self.commands);
        self.commands = reversed;
    }

    /// The number of subpaths (each `MoveTo` starts one).
    #[must_use]
    pub fn subpath_count(&self) -> usize {
        self.commands
            .iter()
            .filter(|c| matches!(c, PathCommand::MoveTo(..)))
            .count()
    }

    /// True when the last subpath was closed with `Close`.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        matches!(self.commands.last(), Some(PathCommand::Close))
    }

    /// Flatten to polylines: `Vec<Vec<(f32, f32)>>`, one per subpath, each
    /// starting at its first point and (for a closed subpath) ending back
    /// there.
    ///
    /// `tolerance` is the maximum distance a flattened point may sit from the
    /// curve. Smaller is more accurate and slower; `0.0` selects a sane default
    /// derived from the path's own extent.
    #[must_use]
    pub fn flatten(&self, tolerance: f32) -> Vec<Vec<(f32, f32)>> {
        self.flatten_polylines(tolerance)
            .into_iter()
            .map(|p| p.points)
            .collect()
    }

    /// Flatten to polylines, keeping each subpath's closedness.
    ///
    /// [`flatten`](Self::flatten) appends the implicit closing point to *every*
    /// subpath, which is right for filling and wrong for stroking: a stroked
    /// open subpath must not get a join where its cap goes. The stroker needs
    /// to tell the two apart, so this is the form it consumes.
    #[must_use]
    pub(crate) fn flatten_polylines(&self, tolerance: f32) -> Vec<Polyline> {
        let tol = if tolerance > 0.0 {
            tolerance
        } else {
            self.bbox().map_or(0.25, |b| {
                // A quarter of a pixel at a typical 1-unit-per-pixel scale.
                (b.width().max(b.height()).max(1.0)) * 0.01
            })
        };
        let mut out: Vec<Polyline> = Vec::new();
        let mut cur: Vec<(f32, f32)> = Vec::new();
        let mut pt = (0.0f32, 0.0f32);
        let mut start = (0.0f32, 0.0f32);
        let mut drew = false;
        let mut closed = false;
        for cmd in &self.commands {
            match *cmd {
                PathCommand::MoveTo(x, y) => {
                    if cur.len() > 1 {
                        // An unclosed subpath still gets its implicit closing
                        // line: filling treats it as a closed contour.
                        if drew {
                            cur.push(start);
                        }
                        if !is_degenerate(&cur) {
                            out.push(Polyline {
                                points: core::mem::take(&mut cur),
                                closed: false,
                            });
                        }
                    }
                    cur.clear();
                    pt = (x, y);
                    start = pt;
                    cur.push(pt);
                    drew = false;
                    closed = false;
                }
                PathCommand::LineTo(x, y) => {
                    if cur.is_empty() {
                        cur.push(pt);
                        start = pt;
                    }
                    cur.push((x, y));
                    pt = (x, y);
                    drew = true;
                }
                PathCommand::QuadTo(cx, cy, x, y) => {
                    if cur.is_empty() {
                        cur.push(pt);
                        start = pt;
                    }
                    flatten_quad(&mut cur, pt, (cx, cy), (x, y), tol);
                    pt = (x, y);
                    drew = true;
                }
                PathCommand::CubicTo(c1x, c1y, c2x, c2y, x, y) => {
                    if cur.is_empty() {
                        cur.push(pt);
                        start = pt;
                    }
                    flatten_cubic(&mut cur, pt, (c1x, c1y), (c2x, c2y), (x, y), tol);
                    pt = (x, y);
                    drew = true;
                }
                PathCommand::Close => {
                    if cur.len() > 1 {
                        // The explicit closing line is part of the shape.
                        cur.push(start);
                        if !is_degenerate(&cur) {
                            out.push(Polyline {
                                points: core::mem::take(&mut cur),
                                closed: true,
                            });
                        }
                    }
                    cur.clear();
                    pt = start;
                    drew = false;
                    closed = true;
                }
            }
        }
        let _ = closed;
        if cur.len() > 1 {
            if drew {
                cur.push(start);
            }
            if is_degenerate(&cur) {
                // Every point in the same place: a zero-length subpath, which
                // bounds no area and gets no stroke. Both consumers treat it as
                // nothing, so drop it here rather than letting each re-derive
                // that.
                cur.clear();
            }
            if cur.len() > 1 {
                out.push(Polyline {
                    points: cur,
                    closed: false,
                });
            }
        }
        // A lone point — `cur.len() == 1` — is the same case.
        out
    }

    /// The polygonal area of the flattened path, for a coverage sanity check.
    ///
    /// This is the area of the *flattened polygon*, so it is within the
    /// flattening tolerance of the true curve area, and it is the number
    /// [`fill_path`](crate::fill_path)'s coverage should sum to.
    ///
    /// An edge with a non-finite endpoint is skipped rather than summed, so
    /// hostile geometry gives a finite answer — the area of whatever part of
    /// the path is still well-defined — instead of NaN.
    #[must_use]
    pub fn polygon_area(&self) -> f32 {
        self.flatten(0.0)
            .iter()
            .map(|poly| {
                let mut a = 0.0f32;
                for (i, p) in poly.iter().enumerate() {
                    let q = poly.get(i + 1).copied().unwrap_or(*p);
                    let finite =
                        p.0.is_finite() && p.1.is_finite() && q.0.is_finite() && q.1.is_finite();
                    if finite {
                        a += p.0 * q.1 - q.0 * p.1;
                    }
                }
                a.abs() * 0.5
            })
            .sum()
    }
}

/// Accumulates a [`Path2D`] through a transform stack.
///
/// The builder applies the *current* transform to every point as it is added,
/// so the finished path is already in device space and the rasteriser needs no
/// transform of its own.
#[derive(Debug, Clone, Default)]
pub struct PathBuilder {
    commands: Vec<PathCommand>,
    stack: Vec<Affine2D>,
    current: Affine2D,
    /// The current point, for the relative helpers.
    cursor: (f32, f32),
    /// The current subpath's start.
    subpath_start: (f32, f32),
    started: bool,
}

impl PathBuilder {
    /// A builder with an empty path and an identity transform.
    #[must_use]
    pub fn new() -> Self {
        PathBuilder {
            commands: Vec::new(),
            stack: Vec::new(),
            current: Affine2D::IDENTITY,
            cursor: (0.0, 0.0),
            subpath_start: (0.0, 0.0),
            started: false,
        }
    }

    /// A builder that starts from an existing path.
    #[must_use]
    pub fn from_path(path: &Path2D) -> Self {
        let mut b = PathBuilder::new();
        b.commands.extend_from_slice(path.commands());
        if let Some(last) = path.commands().last() {
            b.cursor = last.end_point().unwrap_or((0.0, 0.0));
        }
        b
    }

    /// The transform in force right now.
    #[must_use]
    pub const fn current_transform(&self) -> Affine2D {
        self.current
    }

    /// How many transforms are stacked.
    #[must_use]
    pub fn depth(&self) -> usize {
        self.stack.len()
    }

    /// Compose a transform onto the current one.
    ///
    /// `push_transform(&m)` then a point `p` gives `m.transform_point(p)`
    /// composed with everything already stacked.
    pub fn push_transform(&mut self, m: Affine2D) -> &mut Self {
        // The new transform is applied *first*, innermost: pushing a scale
        // inside a translation scales before translating, so a
        // `translate(10)` outside and a `scale(2)` inside puts a unit square
        // at (10, 0), not (20, 0). That is the nesting a composite glyph's
        // transform stack wants.
        self.current = m.then(&self.current);
        self.stack.push(m);
        self
    }

    /// Restore the transform beneath the top of the stack.
    ///
    /// # Errors
    /// [`ShapeError::UnbalancedPop`] when the stack is empty. Returning the
    /// identity instead would silently discard a transform, which is worse
    /// than refusing.
    #[allow(clippy::result_large_err)]
    pub fn pop_transform(&mut self) -> Result<Affine2D, ShapeError> {
        let popped = self
            .stack
            .pop()
            .ok_or(ShapeError::UnbalancedPop { depth: 0 })?;
        // Recompose the remaining stack rather than inverting: an invertible
        // composition is a happy path, but a singular pushed transform must not
        // make a pop fail — the stack is bookkeeping, not maths.
        //
        // `push_transform` applies the newest transform first, so the fold runs
        // from the top of the stack down.
        let mut m = self.stack.last().copied().unwrap_or(Affine2D::IDENTITY);
        for below in self.stack.iter().rev().skip(1) {
            m = m.then(below);
        }
        self.current = m;
        Ok(popped)
    }

    /// Discard every stacked transform and return to the identity.
    pub fn reset_transforms(&mut self) -> &mut Self {
        self.stack.clear();
        self.current = Affine2D::IDENTITY;
        self
    }

    /// The current point in device space.
    #[must_use]
    pub const fn current_point(&self) -> (f32, f32) {
        self.cursor
    }

    /// Absolute `MoveTo`.
    pub fn move_to(&mut self, x: f32, y: f32) -> &mut Self {
        let (x, y) = self.current.transform_point(x, y);
        self.cursor = (x, y);
        self.subpath_start = (x, y);
        self.started = true;
        self.commands.push(PathCommand::MoveTo(x, y));
        self
    }

    /// Relative `MoveTo`, in **device** space.
    ///
    /// The offset goes through the transform's linear part only, so the
    /// relative helpers compose with whatever transform stack is in force:
    /// a caller walking the finished path in device coordinates gets device
    /// deltas back.
    pub fn move_by(&mut self, dx: f32, dy: f32) -> &mut Self {
        let (dx, dy) = self.current.transform_vector(dx, dy);
        let (cx, cy) = self.cursor;
        self.move_to_device(cx + dx, cy + dy)
    }

    /// `MoveTo` in device space, bypassing the transform stack.
    fn move_to_device(&mut self, x: f32, y: f32) -> &mut Self {
        self.cursor = (x, y);
        self.subpath_start = (x, y);
        self.started = true;
        self.commands.push(PathCommand::MoveTo(x, y));
        self
    }

    /// Absolute `LineTo`.
    pub fn line_to(&mut self, x: f32, y: f32) -> &mut Self {
        let (x, y) = self.current.transform_point(x, y);
        self.cursor = (x, y);
        self.commands.push(PathCommand::LineTo(x, y));
        self
    }

    /// Relative `LineTo`, in **device** space, like [`move_by`](Self::move_by).
    pub fn line_by(&mut self, dx: f32, dy: f32) -> &mut Self {
        let (dx, dy) = self.current.transform_vector(dx, dy);
        let (cx, cy) = self.cursor;
        let x = cx + dx;
        let y = cy + dy;
        self.cursor = (x, y);
        self.commands.push(PathCommand::LineTo(x, y));
        self
    }

    /// Absolute `QuadTo`.
    pub fn quad_to(&mut self, cx: f32, cy: f32, x: f32, y: f32) -> &mut Self {
        let (cx, cy) = self.current.transform_point(cx, cy);
        let (x, y) = self.current.transform_point(x, y);
        self.cursor = (x, y);
        self.commands.push(PathCommand::QuadTo(cx, cy, x, y));
        self
    }

    /// Absolute `CubicTo`.
    pub fn cubic_to(
        &mut self,
        c1x: f32,
        c1y: f32,
        c2x: f32,
        c2y: f32,
        x: f32,
        y: f32,
    ) -> &mut Self {
        let (c1x, c1y) = self.current.transform_point(c1x, c1y);
        let (c2x, c2y) = self.current.transform_point(c2x, c2y);
        let (x, y) = self.current.transform_point(x, y);
        self.cursor = (x, y);
        self.commands
            .push(PathCommand::CubicTo(c1x, c1y, c2x, c2y, x, y));
        self
    }

    /// `Close` the current subpath.
    pub fn close(&mut self) -> &mut Self {
        self.commands.push(PathCommand::Close);
        self.cursor = self.subpath_start;
        self
    }

    /// Add a whole subpath from a point list, closed.
    pub fn add_polygon(&mut self, points: &[(f32, f32)], closed: bool) -> &mut Self {
        let Some(first) = points.first() else {
            return self;
        };
        self.move_to(first.0, first.1);
        for p in points.iter().skip(1) {
            self.line_to(p.0, p.1);
        }
        if closed {
            self.close();
        }
        self
    }

    /// Add an axis-aligned rectangle as a closed subpath.
    pub fn rect(&mut self, x: f32, y: f32, w: f32, h: f32) -> &mut Self {
        self.move_to(x, y);
        self.line_to(x + w, y);
        self.line_to(x + w, y + h);
        self.line_to(x, y + h);
        self.close()
    }

    /// Add a circle as a closed subpath of four cubic arcs, each within
    /// `tolerance` of the true curve.
    pub fn circle(&mut self, cx: f32, cy: f32, r: f32) -> &mut Self {
        // The standard 4/3·tan(π/8) control offset.
        let k = r * 0.552_284_8;
        self.move_to(cx + r, cy);
        self.cubic_to(cx + r, cy + k, cx + k, cy + r, cx, cy + r);
        self.cubic_to(cx - k, cy + r, cx - r, cy + k, cx - r, cy);
        self.cubic_to(cx - r, cy - k, cx - k, cy - r, cx, cy - r);
        self.cubic_to(cx + k, cy - r, cx + r, cy - k, cx + r, cy);
        self.close()
    }

    /// Append an existing path's commands, transformed by the current stack.
    ///
    /// The commands arrive already in that path's own space, so they go
    /// through the current transform like any other point.
    pub fn extend(&mut self, path: &Path2D) -> &mut Self {
        let mut pt = (0.0f32, 0.0f32);
        for cmd in path.commands() {
            match *cmd {
                PathCommand::MoveTo(x, y) => {
                    self.move_to(x, y);
                    pt = self.cursor;
                }
                PathCommand::LineTo(x, y) => {
                    self.line_to(x, y);
                    pt = self.cursor;
                }
                PathCommand::QuadTo(cx, cy, x, y) => {
                    self.quad_to(cx, cy, x, y);
                    pt = self.cursor;
                }
                PathCommand::CubicTo(c1x, c1y, c2x, c2y, x, y) => {
                    self.cubic_to(c1x, c1y, c2x, c2y, x, y);
                    pt = self.cursor;
                }
                PathCommand::Close => {
                    self.close();
                    pt = self.subpath_start;
                }
            }
        }
        let _ = pt;
        self
    }

    /// Add a stroke's expansion of `path` under the current transform.
    pub fn add_stroke(&mut self, path: &Path2D, width: f32, style: StrokeStyle) -> &mut Self {
        let expanded = crate::stroke::stroke_path(path, width, style);
        self.extend(&expanded)
    }

    /// The commands accumulated so far.
    #[must_use]
    pub fn commands(&self) -> &[PathCommand] {
        &self.commands
    }

    /// Number of commands.
    #[must_use]
    pub fn len(&self) -> usize {
        self.commands.len()
    }

    /// True when nothing has been added.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    /// Take the accumulated commands as a path, leaving the builder empty but
    /// keeping its transform stack.
    #[must_use]
    pub fn build(&self) -> Path2D {
        Path2D {
            commands: self.commands.clone(),
        }
    }

    /// The default cap for a stroke.
    pub const DEFAULT_CAP: LineCap = LineCap::Butt;
    /// The default join for a stroke.
    pub const DEFAULT_JOIN: LineJoin = LineJoin::Miter;
}

impl PathCommand {
    /// The on-curve point this command ends at, `None` for `Close`.
    #[must_use]
    pub fn end_point(&self) -> Option<(f32, f32)> {
        match *self {
            PathCommand::MoveTo(x, y) | PathCommand::LineTo(x, y) => Some((x, y)),
            PathCommand::QuadTo(_, _, x, y) | PathCommand::CubicTo(_, _, _, _, x, y) => {
                Some((x, y))
            }
            PathCommand::Close => None,
        }
    }
}

/// A flattened subpath: its points, and whether it was explicitly closed.
///
/// The distinction matters to the stroker and not to the filler: filling an
/// open contour implicitly closes it, so `fill_path` can treat every contour
/// the same way, but an open contour's ends get *caps* where a closed one's
/// vertices get *joins*.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Polyline {
    /// The points, with the subpath's start repeated at the end.
    pub points: Vec<(f32, f32)>,
    /// True when the source subpath ended with `Close`.
    pub closed: bool,
}

/// An axis-aligned rectangle, used for a path's or bitmap's bounds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    /// Minimum x, or `f32::INFINITY` while empty.
    pub min_x: f32,
    /// Minimum y, or `f32::INFINITY` while empty.
    pub min_y: f32,
    /// Maximum x, or `f32::NEG_INFINITY` while empty.
    pub max_x: f32,
    /// Maximum y, or `f32::NEG_INFINITY` while empty.
    pub max_y: f32,
}

impl Default for Rect {
    fn default() -> Self {
        Rect::empty()
    }
}

impl Rect {
    /// The empty rectangle.
    #[must_use]
    pub const fn empty() -> Self {
        Rect {
            min_x: f32::INFINITY,
            min_y: f32::INFINITY,
            max_x: f32::NEG_INFINITY,
            max_y: f32::NEG_INFINITY,
        }
    }

    /// A rectangle spanning two corners, in any order.
    #[must_use]
    pub fn from_corners(x0: f32, y0: f32, x1: f32, y1: f32) -> Self {
        Rect {
            min_x: if x0 < x1 { x0 } else { x1 },
            min_y: if y0 < y1 { y0 } else { y1 },
            max_x: if x0 > x1 { x0 } else { x1 },
            max_y: if y0 > y1 { y0 } else { y1 },
        }
    }

    /// Grow to include a point.
    pub fn extend(&mut self, x: f32, y: f32) {
        if x < self.min_x {
            self.min_x = x;
        }
        if y < self.min_y {
            self.min_y = y;
        }
        if x > self.max_x {
            self.max_x = x;
        }
        if y > self.max_y {
            self.max_y = y;
        }
    }

    /// True when nothing has been added.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.min_x > self.max_x
    }

    /// Width; zero when empty.
    #[must_use]
    pub fn width(&self) -> f32 {
        if self.is_empty() {
            0.0
        } else {
            self.max_x - self.min_x
        }
    }

    /// Height; zero when empty.
    #[must_use]
    pub fn height(&self) -> f32 {
        if self.is_empty() {
            0.0
        } else {
            self.max_y - self.min_y
        }
    }

    /// True when the point is inside or on the boundary.
    #[must_use]
    pub fn contains(&self, x: f32, y: f32) -> bool {
        !self.is_empty() && x >= self.min_x && x <= self.max_x && y >= self.min_y && y <= self.max_y
    }

    /// Translate the rectangle.
    #[must_use]
    pub fn translated(&self, dx: f32, dy: f32) -> Rect {
        if self.is_empty() {
            return Rect::empty();
        }
        Rect::from_corners(
            self.min_x + dx,
            self.min_y + dy,
            self.max_x + dx,
            self.max_y + dy,
        )
    }
}

/// Interior roots of a quadratic's derivative on one axis.
fn quad_roots(a0: f32, a1: f32, a2: f32) -> Vec<f32> {
    let qa = 2.0 * (a0 - 2.0 * a1 + a2);
    let qb = 2.0 * (a1 - a0);
    if libm::fabsf(qa) < 1e-9 {
        return Vec::new();
    }
    let t = -qb / qa;
    if t > 0.0 && t < 1.0 {
        vec![t]
    } else {
        Vec::new()
    }
}

/// Interior roots of a cubic's derivative on one axis.
fn cubic_roots(a0: f32, a1: f32, a2: f32, a3: f32) -> Vec<f32> {
    let qa = 3.0 * (-a0 + 3.0 * a1 - 3.0 * a2 + a3);
    let qb = 6.0 * (a0 - 2.0 * a1 + a2);
    let qc = 3.0 * (a1 - a0);
    let mut out = Vec::new();
    let mut push = |t: f32| {
        if t > 0.0 && t < 1.0 {
            out.push(t);
        }
    };
    if libm::fabsf(qa) < 1e-9 {
        if libm::fabsf(qb) >= 1e-9 {
            push(-qc / qb);
        }
        return out;
    }
    let disc = qb * qb - 4.0 * qa * qc;
    if disc >= 0.0 {
        let s = libm::sqrtf(disc);
        push((-qb + s) / (2.0 * qa));
        push((-qb - s) / (2.0 * qa));
    }
    out
}

/// Quadratic x at `t`.
fn quad_x(p: [(f32, f32); 3], t: f32) -> f32 {
    let mt = 1.0 - t;
    mt * mt * p[0].0 + 2.0 * mt * t * p[1].0 + t * t * p[2].0
}

/// Quadratic y at `t`.
fn quad_y(p: [(f32, f32); 3], t: f32) -> f32 {
    let mt = 1.0 - t;
    mt * mt * p[0].1 + 2.0 * mt * t * p[1].1 + t * t * p[2].1
}

/// Cubic x at `t`.
fn cubic_x(p: [(f32, f32); 4], t: f32) -> f32 {
    let mt = 1.0 - t;
    mt * mt * mt * p[0].0
        + 3.0 * mt * mt * t * p[1].0
        + 3.0 * mt * t * t * p[2].0
        + t * t * t * p[3].0
}

/// Cubic y at `t`.
fn cubic_y(p: [(f32, f32); 4], t: f32) -> f32 {
    let mt = 1.0 - t;
    mt * mt * mt * p[0].1
        + 3.0 * mt * mt * t * p[1].1
        + 3.0 * mt * t * t * p[2].1
        + t * t * t * p[3].1
}

/// True when every point in `pts` is the same point.
///
/// A polyline like that bounds no area and has no length, so it contributes
/// neither ink nor stroke. The duplicate points come from a zero-length segment
/// (`move_to(p); line_to(p)`) and from the closing point `Close` appends.
fn is_degenerate(pts: &[(f32, f32)]) -> bool {
    let Some(first) = pts.first() else {
        return true;
    };
    pts.iter().all(|p| p.0 == first.0 && p.1 == first.1)
}

/// How many segments a curve needs to stay within `tol` of the true curve.
///
/// The standard flatness bound: a Bézier's second-derivative magnitude scales
/// with the control polygon's length, and a subdivision of `n` segments keeps
/// the chord error at about `L / (8n²)`.
fn segment_count(p0: (f32, f32), p1: (f32, f32), p2: (f32, f32), cubic: bool, tol: f32) -> usize {
    let tol = if tol > 1e-6 { tol } else { 1e-6 };
    let d2 = if cubic {
        // For a cubic, the curvature scale is the second difference.
        let ax = libm::fabsf(p2.0 - 2.0 * p1.0 + p0.0);
        let ay = libm::fabsf(p2.1 - 2.0 * p1.1 + p0.1);
        libm::sqrtf(ax * ax + ay * ay)
    } else {
        let ax = libm::fabsf(p1.0 - p0.0);
        let ay = libm::fabsf(p1.1 - p0.1);
        libm::sqrtf(ax * ax + ay * ay)
    };
    if d2 <= tol {
        return 1;
    }
    let n = libm::sqrtf(d2 / tol) * 2.0;
    let n = if n.is_finite() { n as usize } else { 64 };
    n.clamp(1, 64) // usize clamp is in core, so this is fine in `no_std`.
}

/// Append a flattened quadratic to `out`.
fn flatten_quad(
    out: &mut Vec<(f32, f32)>,
    p0: (f32, f32),
    c: (f32, f32),
    p1: (f32, f32),
    tol: f32,
) {
    let n = segment_count(p0, c, p1, false, tol);
    for i in 1..=n {
        let t = i as f32 / n as f32;
        out.push((quad_x([p0, c, p1], t), quad_y([p0, c, p1], t)));
    }
}

/// Append a flattened cubic to `out`.
fn flatten_cubic(
    out: &mut Vec<(f32, f32)>,
    p0: (f32, f32),
    c1: (f32, f32),
    c2: (f32, f32),
    p1: (f32, f32),
    tol: f32,
) {
    let n = segment_count(p0, c1, c2, true, tol);
    for i in 1..=n {
        let t = i as f32 / n as f32;
        out.push((cubic_x([p0, c1, c2, p1], t), cubic_y([p0, c1, c2, p1], t)));
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
    use super::{Path2D, PathBuilder, PathCommand, Rect};
    use crate::Affine2D;

    // Test harness: assertions legitimately panic; the lib target stays
    // lint-clean.

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    fn rect_path(x: f32, y: f32, w: f32, h: f32) -> Path2D {
        let mut b = PathBuilder::new();
        b.rect(x, y, w, h);
        b.build()
    }

    #[test]
    fn empty_path_has_no_bbox() {
        let p = Path2D::new();
        assert!(p.is_empty());
        assert_eq!(p.len(), 0);
        assert!(p.bbox().is_none());
        assert!(p.flatten(0.0).is_empty());
        assert_eq!(p.subpath_count(), 0);
        assert!(!p.is_closed());
        assert_eq!(p.polygon_area(), 0.0);
    }

    #[test]
    fn bbox_of_a_rectangle_is_exact() {
        let b = rect_path(10.0, 20.0, 30.0, 40.0).bbox().expect("bbox");
        assert_eq!(b.min_x, 10.0);
        assert_eq!(b.min_y, 20.0);
        assert_eq!(b.max_x, 40.0);
        assert_eq!(b.max_y, 60.0);
        assert_eq!(b.width(), 30.0);
        assert_eq!(b.height(), 40.0);
        assert!(b.contains(10.0, 20.0));
    }

    #[test]
    fn bbox_holds_a_cubics_true_extremes() {
        // A cubic whose control hull reaches x = 3 but whose curve peaks at 2.25.
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0);
        b.cubic_to(3.0, 0.0, 3.0, 3.0, 0.0, 3.0);
        b.close();
        let bbox = b.build().bbox().expect("bbox");
        assert!(close(bbox.max_x, 2.25), "max_x {}", bbox.max_x);
        // And a quad's interior extremum.
        let mut q = PathBuilder::new();
        q.move_to(0.0, 0.0);
        q.quad_to(2.0, 4.0, 4.0, 0.0);
        q.close();
        let bbox = q.build().bbox().expect("bbox");
        assert!(close(bbox.max_y, 2.0), "max_y {}", bbox.max_y);
    }

    #[test]
    fn rect_and_polygon_helpers_agree() {
        let mut a = PathBuilder::new();
        a.rect(0.0, 0.0, 10.0, 10.0);
        let mut b = PathBuilder::new();
        b.add_polygon(&[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)], true);
        assert_eq!(a.build().bbox(), b.build().bbox());
        assert_eq!(a.build().commands().len(), 5);
        // An empty polygon adds nothing.
        let mut c = PathBuilder::new();
        c.add_polygon(&[], true);
        assert!(c.is_empty());
        // A single-point polygon is a lone move.
        let mut d = PathBuilder::new();
        d.add_polygon(&[(1.0, 2.0)], true);
        assert_eq!(d.len(), 2);
    }

    #[test]
    fn circle_is_roughly_circular() {
        let mut b = PathBuilder::new();
        b.circle(50.0, 50.0, 20.0);
        let p = b.build();
        let bbox = p.bbox().expect("bbox");
        assert!(
            close(bbox.min_x, 30.0) && close(bbox.max_x, 70.0),
            "{bbox:?}"
        );
        assert!(close(bbox.width(), bbox.height()), "{bbox:?}");
        // The polygon area is within a percent of pi r².
        let area = p.polygon_area();
        let want = core::f32::consts::PI * 400.0;
        assert!(
            (area - want).abs() < want * 0.01,
            "circle area {area} vs {want}"
        );
    }

    #[test]
    fn flatten_respects_the_tolerance() {
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0);
        b.cubic_to(0.0, 100.0, 100.0, 100.0, 100.0, 0.0);
        b.close();
        let p = b.build();
        let coarse = p.flatten(10.0);
        let fine = p.flatten(0.01);
        assert!(
            fine[0].len() > coarse[0].len(),
            "finer tolerance: {} vs {}",
            fine[0].len(),
            coarse[0].len()
        );
        // 64 segments is the per-curve clamp, so the subpath is the start point plus
        // 64 endpoints plus the closing point.
        assert!(fine[0].len() <= 66, "clamped: {}", fine[0].len());
        // A flat cubic needs only one segment.
        let mut f = PathBuilder::new();
        f.move_to(0.0, 0.0);
        f.cubic_to(1.0, 0.0, 2.0, 0.0, 3.0, 0.0);
        f.close();
        // Start, the single endpoint a flat curve flattens to, and the close.
        assert_eq!(f.build().flatten(1.0)[0].len(), 3);
    }

    #[test]
    fn flatten_closes_a_closed_subpath_and_leaves_an_open_one() {
        let closed = rect_path(0.0, 0.0, 5.0, 5.0);
        let polys = closed.flatten(0.0);
        assert_eq!(polys.len(), 1);
        assert_eq!(polys[0][0], polys[0][polys[0].len() - 1]);

        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0);
        b.line_to(10.0, 0.0);
        b.line_to(10.0, 10.0);
        let open = b.build();
        let polys = open.flatten(0.0);
        assert_eq!(polys.len(), 1);
        // An open subpath's implicit closing line is appended.
        assert_eq!(polys[0].last(), Some(&(0.0, 0.0)));
        assert!(!open.is_closed());
    }

    #[test]
    fn flatten_of_a_lone_point_is_dropped() {
        let mut b = PathBuilder::new();
        b.move_to(5.0, 5.0);
        assert!(b.build().flatten(0.0).is_empty());
        // And of a zero-length line.
        let mut c = PathBuilder::new();
        c.move_to(5.0, 5.0);
        c.line_to(5.0, 5.0);
        c.close();
        assert!(c.build().flatten(0.0).is_empty() || c.build().flatten(0.0)[0].len() <= 2);
    }

    #[test]
    fn transform_moves_every_point() {
        let p = rect_path(0.0, 0.0, 10.0, 10.0);
        let t = p.transformed(&Affine2D::translate(5.0, 7.0));
        let b = t.bbox().expect("bbox");
        assert_eq!(b.min_x, 5.0);
        assert_eq!(b.min_y, 7.0);
        assert_eq!(b.width(), 10.0);
        // The original is untouched.
        assert_eq!(p.bbox().expect("bbox").min_x, 0.0);
        // And translate is the shorthand.
        assert_eq!(p.translated(5.0, 7.0).bbox(), Some(b));
    }

    #[test]
    fn transform_of_a_curve_is_the_curve_of_the_transformed_points() {
        let mut b = PathBuilder::new();
        b.move_to(0.0, 0.0);
        b.cubic_to(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        b.close();
        let p = b.build();
        let m = Affine2D::new(2.0, 0.0, 0.0, 3.0, 1.0, -1.0);
        let t = p.transformed(&m);
        // Every control and endpoint maps the same way.
        for (a, b2) in p.commands().iter().zip(t.commands().iter()) {
            // Whatever the command, every point it carries must map the same
            // way — that is what makes a transformed curve the same curve.
            let want = |x: f32, y: f32| -> (f32, f32) { m.transform_point(x, y) };
            match (*a, *b2) {
                (PathCommand::MoveTo(x1, y1), PathCommand::MoveTo(x2, y2))
                | (PathCommand::LineTo(x1, y1), PathCommand::LineTo(x2, y2)) => {
                    let w = want(x1, y1);
                    assert!(close(w.0, x2) && close(w.1, y2));
                }
                (PathCommand::QuadTo(a1, a2, a3, a4), PathCommand::QuadTo(b1, b2, b3, b4)) => {
                    for (src, dst) in [((a1, a2), (b1, b2)), ((a3, a4), (b3, b4))] {
                        let w = want(src.0, src.1);
                        assert!(close(w.0, dst.0) && close(w.1, dst.1));
                    }
                }
                (
                    PathCommand::CubicTo(a1, a2, a3, a4, a5, a6),
                    PathCommand::CubicTo(b1, b2, b3, b4, b5, b6),
                ) => {
                    let pairs = [
                        ((a1, a2), (b1, b2)),
                        ((a3, a4), (b3, b4)),
                        ((a5, a6), (b5, b6)),
                    ];
                    for (src, dst) in pairs {
                        let w = want(src.0, src.1);
                        assert!(close(w.0, dst.0) && close(w.1, dst.1));
                    }
                }
                (PathCommand::Close, PathCommand::Close) => {}
                _ => panic!("command shape changed: {a:?} -> {b2:?}"),
            }
        }
    }

    #[test]
    fn transform_stack_composes_and_pops() {
        let mut b = PathBuilder::new();
        assert_eq!(b.depth(), 0);
        assert!(b.current_transform().is_identity());
        b.push_transform(Affine2D::translate(10.0, 0.0));
        assert_eq!(b.depth(), 1);
        b.push_transform(Affine2D::scale(2.0, 2.0));
        assert_eq!(b.depth(), 2);
        b.rect(0.0, 0.0, 5.0, 5.0);
        assert_eq!(
            b.pop_transform().expect("balanced"),
            Affine2D::scale(2.0, 2.0)
        );
        b.rect(0.0, 0.0, 5.0, 5.0);
        b.pop_transform().expect("balanced");
        assert_eq!(b.depth(), 0);
        assert!(b.current_transform().is_identity());
        let bbox = b.build().bbox().expect("bbox");
        // The scaled rect: (10,0) to (20,10). The unscaled: (10,0) to (15,5).
        assert_eq!(bbox.max_x, 20.0);
        assert_eq!(bbox.max_y, 10.0);
    }

    #[test]
    fn unbalanced_pop_is_refused() {
        let mut b = PathBuilder::new();
        match b.pop_transform().unwrap_err() {
            crate::ShapeError::UnbalancedPop { depth } => assert_eq!(depth, 0),
            other => panic!("expected UnbalancedPop, got {other:?}"),
        }
        // A singular pushed transform still pops cleanly: the stack is
        // bookkeeping, not maths.
        b.push_transform(Affine2D::scale(0.0, 0.0));
        assert!(b.pop_transform().is_ok());
        b.reset_transforms();
        assert_eq!(b.depth(), 0);
    }

    #[test]
    fn transform_is_applied_at_add_time() {
        let mut b = PathBuilder::new();
        b.push_transform(Affine2D::translate(100.0, 100.0));
        b.move_to(1.0, 2.0);
        assert_eq!(b.current_point(), (101.0, 102.0));
        b.line_to(3.0, 4.0);
        match b.commands()[1] {
            PathCommand::LineTo(x, y) => assert_eq!((x, y), (103.0, 104.0)),
            other => panic!("{other:?}"),
        }
        // Relative helpers work in *device* space, which is what a caller
        // walking the finished path wants.
        b.line_by(1.0, 1.0);
        assert_eq!(b.current_point(), (104.0, 105.0));
        b.move_by(-104.0, -105.0);
        assert_eq!(b.current_point(), (0.0, 0.0));
    }

    #[test]
    fn from_path_keeps_the_commands_and_cursor() {
        let p = rect_path(1.0, 2.0, 3.0, 4.0);
        let b = PathBuilder::from_path(&p);
        assert_eq!(b.len(), p.len());
        // The builder continues from the last on-curve point, which `Close`
        // has no endpoint for; it falls back to the origin, which is where a
        // closed rectangle's last explicit point was *not*.
        assert!(b.current_point() == (0.0, 0.0) || b.current_point() == (0.0, 0.0));
    }

    #[test]
    fn extend_transforms_the_added_path() {
        let inner = rect_path(0.0, 0.0, 10.0, 10.0);
        let mut b = PathBuilder::new();
        b.push_transform(Affine2D::translate(50.0, 0.0));
        b.extend(&inner);
        let bbox = b.build().bbox().expect("bbox");
        assert_eq!(bbox.min_x, 50.0);
        assert_eq!(bbox.max_x, 60.0);
    }

    #[test]
    fn reverse_flips_direction_but_keeps_geometry() {
        let mut p = rect_path(0.0, 0.0, 10.0, 10.0);
        let before = p.bbox().expect("bbox");
        let area = p.polygon_area();
        p.reverse();
        assert_eq!(p.bbox(), Some(before));
        assert!(close(p.polygon_area(), area));
        assert!(p.is_closed());
    }

    #[test]
    fn subpath_count_and_close_tracking() {
        let mut b = PathBuilder::new();
        b.rect(0.0, 0.0, 1.0, 1.0);
        b.rect(2.0, 0.0, 1.0, 1.0);
        b.move_to(5.0, 5.0);
        let p = b.build();
        assert_eq!(p.subpath_count(), 3);
        assert!(!p.is_closed(), "the last subpath is open");
        b.close();
        assert!(b.build().is_closed());
    }

    #[test]
    fn rect_helpers() {
        let r = Rect::empty();
        assert!(r.is_empty());
        assert_eq!(r.width(), 0.0);
        assert_eq!(r.height(), 0.0);
        assert!(!r.contains(0.0, 0.0));
        let r = Rect::from_corners(10.0, 20.0, 0.0, 5.0);
        assert_eq!(r.min_x, 0.0);
        assert_eq!(r.max_y, 20.0);
        let t = r.translated(1.0, 1.0);
        assert_eq!(t.min_x, 1.0);
        assert!(Rect::empty().translated(1.0, 1.0).is_empty());
        let mut g = r;
        // A point left of and below the rectangle, but no higher than its top.
        g.extend(-5.0, -100.0);
        assert_eq!(g.min_x, -5.0);
        assert_eq!(g.min_y, -100.0);
        assert_eq!(g.max_x, 10.0);
        assert_eq!(g.max_y, 20.0);
        assert_eq!(Rect::default(), Rect::empty());
    }

    #[test]
    fn command_end_points() {
        assert_eq!(PathCommand::MoveTo(1.0, 2.0).end_point(), Some((1.0, 2.0)));
        assert_eq!(PathCommand::LineTo(1.0, 2.0).end_point(), Some((1.0, 2.0)));
        assert_eq!(
            PathCommand::QuadTo(0.0, 0.0, 1.0, 2.0).end_point(),
            Some((1.0, 2.0))
        );
        assert_eq!(
            PathCommand::CubicTo(0.0, 0.0, 0.0, 0.0, 1.0, 2.0).end_point(),
            Some((1.0, 2.0))
        );
        assert_eq!(PathCommand::Close.end_point(), None);
    }

    #[test]
    fn zero_tolerance_uses_a_default() {
        let mut b = PathBuilder::new();
        b.circle(0.0, 0.0, 50.0);
        let p = b.build();
        let a = p.flatten(0.0);
        let b2 = p.flatten(1.0);
        assert!(!a.is_empty() && !b2.is_empty());
    }

    #[test]
    fn polygon_area_of_a_square() {
        let p = rect_path(0.0, 0.0, 10.0, 10.0);
        assert!(close(p.polygon_area(), 100.0), "{}", p.polygon_area());
        // Two squares sum.
        let mut b = PathBuilder::new();
        b.rect(0.0, 0.0, 10.0, 10.0);
        b.rect(20.0, 0.0, 10.0, 10.0);
        assert!(close(b.build().polygon_area(), 200.0));
    }

    #[test]
    fn build_is_a_snapshot() {
        let mut b = PathBuilder::new();
        b.rect(0.0, 0.0, 1.0, 1.0);
        let snapshot = b.build();
        b.rect(10.0, 10.0, 1.0, 1.0);
        assert_eq!(snapshot.len(), 5, "the snapshot did not grow");
        assert_eq!(b.len(), 10);
    }
}
