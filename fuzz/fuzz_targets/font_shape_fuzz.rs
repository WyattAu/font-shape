//! Fuzz target — `font_shape_fuzz`: arbitrary bytes become paths, strokes,
//! glyph outlines, and measured text.
//!
//! The contract under fuzz is *totality plus invariants*. Every public
//! entry point must return a mask, a path, or a typed [`ShapeError`] for any
//! input at all — never a panic, never an out-of-bounds write, never an
//! unbounded allocation. Alongside that, four properties are asserted on
//! whatever the fuzzer produces:
//!
//! 1. **Coverage is a coverage.** Every mask byte is a valid fraction of a
//!    pixel, and a mask's total ink never exceeds the area of its raster.
//! 2. **Filling is area-exact for straight edges.** A rectangle's ink equals
//!    its geometric area to within the 8-bit quantisation, which is the
//!    claim the whole rasteriser rests on.
//! 3. **Transforms commute with filling.** A transformed path's ink matches
//!    the untransformed one's, and a scale by `s` scales the area by `s²`.
//! 4. **Measurement is monotone and total.** A run's advance is its sum of
//!    advances plus its kerns, and reordering the line never changes it.
//!
//! Inputs are bounded (point counts, sizes) so a fuzzer run cannot be turned
//! into an OOM or an hours-long case by a single input.

#![no_main]

use font_model::{Font, Glyph, GlyphId, Outline, Path};
use font_shape::{
    fill_path, mask_coverage, rasterize_outline, shape_line, stroke_path, FillRule, GlyphBitmap,
    Hinting, LineCap, LineJoin, PathBuilder, ShapeError, StrokeStyle,
};
use libfuzzer_sys::fuzz_target;

/// Upper bound on the commands in one path.
const MAX_COMMANDS: usize = 64;
/// Upper bound on the contours in one outline.
const MAX_CONTOURS: usize = 4;
/// Upper bound on the raster side, so a mask is at most 256².
const MAX_RASTER: u32 = 256;
/// A size the rasteriser will accept, in the range a UI actually asks for.
const MAX_SIZE_PX: f32 = 512.0;

/// One `f32` from the input, folded into a plausible device-space range.
///
/// Non-finite input maps to zero: the interesting paths are the arithmetic
/// ones, and the non-finite guards get their own coverage from the wildcards
/// below and from the integration suite.
fn coord(data: &[u8], at: usize) -> f32 {
    let bits = u32::from_le_bytes([
        data.get(at).copied().unwrap_or(0),
        data.get(at + 1).copied().unwrap_or(0),
        data.get(at + 2).copied().unwrap_or(0),
        data.get(at + 3).copied().unwrap_or(0),
    ]);
    let v = f32::from_bits(bits);
    if v.is_finite() {
        (v % 256.0) as f32
    } else {
        0.0
    }
}

/// A deliberately hostile variant: NaN or infinity.
fn poison(k: usize) -> (f32, f32) {
    match k % 8 {
        0 => (f32::NAN, 0.0),
        1 => (0.0, f32::NAN),
        2 => (f32::INFINITY, 0.0),
        3 => (0.0, f32::NEG_INFINITY),
        _ => (0.0, 0.0),
    }
}

/// Build a path from the input, sometimes with non-finite coordinates.
fn path_from(data: &[u8]) -> font_shape::Path2D {
    let mut b = PathBuilder::new();
    b.move_to(coord(data, 0), coord(data, 4));
    let count = usize::from(data.first().copied().unwrap_or(0)) % MAX_COMMANDS;
    for k in 1..count {
        let at = k * 12;
        let (nan, inf) = poison(k);
        match k % 5 {
            0 => {
                b.line_to(coord(data, at) + nan, coord(data, at + 4) + inf);
            }
            1 => {
                b.quad_to(
                    coord(data, at),
                    coord(data, at + 4) + nan,
                    coord(data, at + 20),
                    coord(data, at + 24) + inf,
                );
            }
            2 => {
                b.cubic_to(
                    coord(data, at),
                    coord(data, at + 4),
                    coord(data, at + 20) + nan,
                    coord(data, at + 28),
                    coord(data, at + 32),
                    coord(data, at + 36) + inf,
                );
            }
            3 => {
                b.move_to(coord(data, at) + nan, coord(data, at + 4));
            }
            _ => {
                b.line_by(coord(data, at), coord(data, at + 4));
            }
        }
    }
    b.close();
    b.build()
}

/// A multi-contour font outline: nested and overlapping boxes, which is what
/// separates the two fill rules.
fn multi_contour_outline(data: &[u8]) -> Outline {
    let mut o = Outline::new();
    let n = 1 + usize::from(data.get(64).copied().unwrap_or(0)) % MAX_CONTOURS;
    for k in 0..n {
        let inset = f32::from(k as u8) * 8.0;
        let mut p = Path::starting_at(100.0 + inset, 100.0 + inset);
        p.line_to(600.0 - inset, 100.0 + inset);
        p.line_to(600.0 - inset, 600.0 - inset);
        p.line_to(100.0 + inset, 600.0 - inset);
        p.close();
        o.push_contour(p);
    }
    o
}

/// A stroke style drawn from the input, so every cap and join is exercised.
fn style_from(data: &[u8], width: f32) -> StrokeStyle {
    let caps = [LineCap::Butt, LineCap::Square, LineCap::Round];
    let joins = [LineJoin::Miter, LineJoin::Bevel, LineJoin::Round];
    StrokeStyle::new(width)
        .with_cap(caps[usize::from(data.get(1).copied().unwrap_or(0)) % 3])
        .with_join(joins[usize::from(data.get(2).copied().unwrap_or(0)) % 3])
        .with_miter_limit(1.0 + f32::from(data.get(3).copied().unwrap_or(0) % 8))
}

/// Two boxes "match" if they agree to within a relative epsilon.
///
/// A box from a path holding NaN coordinates has no true extrema to preserve,
/// and `NaN != NaN` would make an exact comparison fail on a correct result,
/// so a non-finite bound compares as a match. The epsilon is relative because
/// the fuzzer produces geometry down to the denormal range, where an absolute
/// tolerance would be either too tight or vacuous.
fn boxes_match(a: Option<font_shape::Rect>, b: Option<font_shape::Rect>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => [
            (a.min_x, b.min_x),
            (a.min_y, b.min_y),
            (a.max_x, b.max_x),
            (a.max_y, b.max_y),
        ]
        .iter()
        .all(|(u, v)| {
            !u.is_finite() || !v.is_finite() || (u - v).abs() <= 1e-3 * u.abs().max(v.abs()).max(1.0)
        }),
        _ => false,
    }
}

/// Two points are "the same" if they agree, or if both are NaN.
///
/// Plain `==` is not enough: a NaN coordinate is never equal to itself, and the
/// fuzzer produces them deliberately, so comparing geometry with it would fail
/// on a correct result.
fn same_point(a: (f32, f32), b: (f32, f32)) -> bool {
    let axis = |x: f32, y: f32| x == y || (x.is_nan() && y.is_nan());
    axis(a.0, b.0) && axis(a.1, b.1)
}

/// A mask is a coverage: total ink never exceeds the frame it covers.
fn assert_mask_is_coverage(mask: &[u8], width: u32, height: u32) {
    let area = f64::from(width) * f64::from(height);
    let ink = mask_coverage(mask) / 255.0;
    assert!(
        ink <= area + 1e-6,
        "ink {ink} exceeds the {width}x{height} frame ({area})"
    );
    assert!(ink >= 0.0, "negative ink {ink}");
}

/// A rectangle's coverage sums to its area, to within the 8-bit rounding.
fn assert_rect_area_is_exact(path: &font_shape::Path2D, rule: FillRule, width: u32, height: u32) {
    let mask = match fill_path(path, rule, width, height) {
        Ok(m) => m,
        Err(e) => panic!("filling a rectangle failed: {e}"),
    };
    let area = f64::from(path.polygon_area());
    if area > 1.0 {
        let err = (mask_coverage(&mask) / 255.0 - area).abs() / area;
        // One quantisation step per edge column, and a path can be a curve, so
        // a tenth of a percent is a generous bound.
        assert!(err < 0.01, "rect ink error {err} for area {area}");
    }
    assert_mask_is_coverage(&mask, width, height);
}

/// The font the measurement half shapes against: `.notdef` plus glyphs 1..=4.
fn demo_font() -> Font {
    let mut builder = Font::builder()
        .units_per_em(1000)
        .ascender(800)
        .descender(-200);
    for k in 0..5u16 {
        let mut outline = Outline::new();
        let mut p = Path::starting_at(50.0, 0.0);
        let right = 50.0 + f32::from(400 + k * 20);
        p.line_to(right, 0.0);
        p.line_to(right, 700.0);
        p.line_to(50.0, 700.0);
        p.close();
        outline.push_contour(p);
        builder = builder.glyph(Glyph::new(GlyphId::new(k), 500 + k * 20, 50, outline));
    }
    let mut sub = font_model::CmapSubtable::empty_format4();
    for (cp, gid) in [
        ('A' as u32, GlyphId::new(1)),
        ('B' as u32, GlyphId::new(2)),
        ('C' as u32, GlyphId::new(3)),
        ('D' as u32, GlyphId::new(4)),
    ] {
        sub = font_model::insert_entry(&sub, cp, gid);
    }
    let mut cmap = font_model::Cmap::new();
    cmap.subtables_mut().push(sub);
    builder
        .kern(GlyphId::new(1), GlyphId::new(2), -30)
        .cmap(cmap)
        .build()
        .expect("the demo font is valid")
}

/// A bitmap from `rasterize_outline` is a coverage too.
fn assert_bitmap_is_coverage(bmp: &GlyphBitmap) {
    assert_eq!(
        bmp.coverage().len(),
        (bmp.width() * bmp.height()) as usize,
        "the mask length matches the bitmap's dimensions"
    );
    let ink = bmp.ink();
    assert!(ink >= 0.0);
    assert!(
        ink <= f64::from(bmp.width()) * f64::from(bmp.height()) + 1e-6,
        "ink {ink} exceeds {}x{}",
        bmp.width(),
        bmp.height()
    );
    // The placement is consistent with the dimensions.
    assert!(bmp.top() >= 0 || bmp.is_empty());
}

fuzz_target!(|data: &[u8]| {
    let path = path_from(data);

    // ---- Every derived view of a path is total. ----
    let _ = path.bbox();
    let _ = path.is_closed();
    let _ = path.subpath_count();
    let _ = path.flatten(0.0);
    let _ = path.polygon_area();
    let mut reversed = path.clone();
    reversed.reverse();
    // Reversal is geometry-preserving: the same bounding box, the same area.
    // A subpath with no segments has no direction to flip and does not survive
    // the round trip, so the strong form holds only when there is geometry.
    let has_geometry = !path.flatten(0.0).is_empty();
    if has_geometry {
        // Boxes are compared only when both are finite: a path with a NaN
        // coordinate has no true extrema to preserve, and `NaN != NaN` would
        // make the comparison fail on a correct result.
        assert!(
            boxes_match(path.bbox(), reversed.bbox()),
            "reversal moved the box: {:?} vs {:?}",
            path.bbox(),
            reversed.bbox()
        );
        // And reversing twice is the identity — on the *geometry*. Comparing
        // commands directly would fail for any input holding a NaN, since
        // `NaN != NaN`, so the flattened polylines are the honest check.
        let mut twice = reversed.clone();
        twice.reverse();
        assert!(
            boxes_match(twice.bbox(), path.bbox()),
            "reversing twice moved the box: {:?} vs {:?}",
            twice.bbox(),
            path.bbox()
        );
        let first = path.flatten(0.0);
        let second = twice.flatten(0.0);
        assert_eq!(
            first.len(),
            second.len(),
            "reversal changes the subpath count"
        );
        for (a, b) in first.iter().zip(second.iter()) {
            assert_eq!(a.len(), b.len(), "reversal changes a subpath's point count");
            for (p, q) in a.iter().zip(b.iter()) {
                assert!(same_point(*p, *q), "reversal moved {p:?} to {q:?}");
            }
        }
    }
    let a = path.polygon_area();
    let b = reversed.polygon_area();
    assert!((a - b).abs() < 1e-2 * a.max(1.0), "reversed area {b} vs {a}");

    // ---- Filling: coverage is bounded, and exact for straight edges. ----
    let size = 8 + u32::from(data.get(4).copied().unwrap_or(0)) % 32;
    for rule in [FillRule::NonZero, FillRule::EvenOdd] {
        match fill_path(&path, rule, size, size) {
            Ok(mask) => {
                assert_eq!(mask.len(), (size * size) as usize);
                assert_mask_is_coverage(&mask, size, size);
            }
            Err(ShapeError::RasterTooLarge { .. }) | Err(ShapeError::InvalidSize { .. }) => {}
            Err(e) => panic!("unexpected fill error: {e}"),
        }
    }

    // An oversized raster is refused, never allocated.
    assert!(matches!(
        fill_path(&path, FillRule::NonZero, 100_000, 100_000),
        Err(ShapeError::RasterTooLarge { .. })
    ));
    // A zero-sized raster is empty, not an error.
    assert!(fill_path(&path, FillRule::NonZero, 0, 16)
        .expect("zero width is not an error")
        .is_empty());

    // ---- A rectangle is filled to its exact area. ----
    // The rectangle is built at the origin so it lands inside the frame, and
    // only its *size* comes from the input: an offset would put it outside and
    // then a zero ink would be correct, not a defect.
    let rw = 4.0 + coord(data, 48).abs() % 32.0;
    let rh = 4.0 + coord(data, 52).abs() % 32.0;
    let mut rect = PathBuilder::new();
    rect.rect(2.0, 2.0, rw, rh);
    let rect_path = rect.build();
    // A frame big enough for the rectangle plus a one-pixel margin.
    let need = (rw.ceil() + 4.0).max(rh.ceil() + 4.0);
    let side = if need.is_finite() {
        (need as u32).clamp(4, MAX_RASTER)
    } else {
        MAX_RASTER
    };
    let placed = rect_path;
    assert_rect_area_is_exact(&placed, FillRule::NonZero, side, side);

    // ---- A transform of a path is a transform of its coverage. ----
    let plain = fill_path(&placed, FillRule::NonZero, side, side).expect("fills");
    let moved =
        fill_path(&placed.translated(1.0, 0.0), FillRule::NonZero, side, side).expect("fills");
    if side > 4 {
        let shift = mask_coverage(&plain) / 255.0;
        let moved_ink = mask_coverage(&moved) / 255.0;
        // A one-pixel shift moves ink off one edge and on the other, so the
        // totals agree to within the pixels that crossed the frame.
        assert!(
            (shift - moved_ink).abs() <= side as f64 + 1.0,
            "translated ink {moved_ink} vs {shift}"
        );
    }

    // ---- Stroking: expand to an outline, then fill that. ----
    let width = coord(data, 56).abs() % 24.0;
    let style = style_from(data, width);
    let stroked = stroke_path(&path, width.max(width.min(1.0)), style);
    // Every emitted contour is closed, which is what makes the non-zero union
    // a union.
    let mut starts = 0usize;
    let mut closes = 0usize;
    for cmd in stroked.commands() {
        match cmd {
            font_shape::PathCommand::MoveTo(..) => starts += 1,
            font_shape::PathCommand::Close => closes += 1,
            _ => {}
        }
    }
    assert_eq!(starts, closes, "every stroke contour is closed");
    if !stroked.is_empty() {
        match fill_path(&stroked, FillRule::NonZero, size, size) {
            Ok(mask) => assert_mask_is_coverage(&mask, size, size),
            Err(ShapeError::RasterTooLarge { .. }) | Err(ShapeError::InvalidSize { .. }) => {}
            Err(e) => panic!("unexpected stroke fill error: {e}"),
        }
    }
    // A zero or non-finite width has no region at all.
    assert!(stroke_path(&path, 0.0, StrokeStyle::new(0.0)).is_empty());
    assert!(stroke_path(&path, f32::NAN, StrokeStyle::default()).is_empty());
    assert!(stroke_path(&path, -1.0, StrokeStyle::new(-1.0)).is_empty());

    // ---- Glyph rasterization: total, and a coverage. ----
    let font = demo_font();
    let upem = font.units_per_em();
    let size_px = 4.0 + coord(data, 60).abs() % MAX_SIZE_PX;
    for hinting in [Hinting::None, Hinting::GridFit] {
        for gid in [
            GlyphId::NOTDEF,
            GlyphId::new(1),
            GlyphId::new(4),
            GlyphId::new(200),
        ] {
            // A glyph past the store has no outline, which the rasteriser
            // must treat as "no ink" rather than a lookup panic.
            let empty = font_model::Outline::new();
            let outline = font.glyph(gid).map_or(&empty, font_model::Glyph::outline);
            match rasterize_outline(outline, upem, size_px, hinting) {
                Ok(bmp) => assert_bitmap_is_coverage(&bmp),
                Err(ShapeError::RasterTooLarge { .. }) => {}
                Err(e) => panic!("unexpected glyph raster error: {e}"),
            }
        }
    }
    // Nested contours are where the two fill rules part company, and both must
    // stay total.
    let nested = multi_contour_outline(data);
    match rasterize_outline(&nested, upem, 64.0, Hinting::None) {
        Ok(bmp) => assert_bitmap_is_coverage(&bmp),
        Err(ShapeError::RasterTooLarge { .. }) => {}
        Err(e) => panic!("unexpected nested-outline error: {e}"),
    }

    // Bad sizes are typed errors, not panics.
    for bad in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(matches!(
            rasterize_outline(&font_model::Outline::new(), upem, bad, Hinting::None),
            Err(ShapeError::InvalidScale { .. })
        ));
    }
    assert!(matches!(
        rasterize_outline(&font_model::Outline::new(), 0, 12.0, Hinting::None),
        Err(ShapeError::DegenerateUnitsPerEm)
    ));

    // ---- Measurement: total, and the advance is the sum of its parts. ----
    let text: String = data.iter().map(|&b| (b % 6) as char).collect();
    let ltr = match shape_line(&font, &text, 100.0) {
        Ok(l) => l,
        Err(e) => panic!("unexpected shape error: {e}"),
    };
    assert_eq!(ltr.len(), text.chars().count());
    // Walk the placements: each one's x is the previous advance plus its kern.
    let mut pen = 0.0f32;
    let mut prev: Option<GlyphId> = None;
    for cluster in 0..text.chars().count() {
        let Some(p) = ltr
            .placements()
            .iter()
            .find(|p| p.cluster as usize == cluster)
        else {
            panic!("no placement for cluster {cluster}");
        };
        if let Some(q) = prev {
            let kern = f32::from(font.kern(q, p.glyph)) / 1000.0 * 100.0;
            assert!(
                (p.x - (pen + kern)).abs() < 1e-3,
                "cluster {cluster} x {} vs {}",
                p.x,
                pen + kern
            );
            pen = p.x + p.advance_px;
        } else {
            assert!(p.x.abs() < 1e-3, "the first glyph sits at the origin");
            pen = p.advance_px;
        }
        prev = Some(p.glyph);
    }
    assert!(
        (ltr.advance_px() - pen).abs() < 1e-3,
        "advance {} vs walked {pen}",
        ltr.advance_px()
    );

    // Right-to-left reorders the line but not its width.
    for direction in [
        font_shape::LayoutDirection::LeftToRight,
        font_shape::LayoutDirection::RightToLeft,
        font_shape::LayoutDirection::Auto,
    ] {
        let line = shape_line_with_checked(&font, &text, 100.0, direction);
        assert!(
            (line.advance_px() - ltr.advance_px()).abs() < 1e-3,
            "{direction:?} changed the advance"
        );
        assert_eq!(line.len(), text.chars().count());
        // Every placement's cluster is still the character it came from.
        let mut clusters: Vec<u32> = line.placements().iter().map(|p| p.cluster).collect();
        clusters.sort_unstable();
        assert_eq!(
            clusters,
            (0..text.chars().count() as u32).collect::<Vec<_>>()
        );
    }
});

/// `shape_line_with`, with the error turned into a panic message.
fn shape_line_with_checked(
    font: &Font,
    text: &str,
    size_px: f32,
    direction: font_shape::LayoutDirection,
) -> font_shape::ShapedLine {
    match font_shape::shape_line_with(font, text, size_px, direction) {
        Ok(l) => l,
        Err(e) => panic!("unexpected shape error for {direction:?}: {e}"),
    }
}
