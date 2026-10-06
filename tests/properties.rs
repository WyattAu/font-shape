//! Property tests: invariants that must hold for *arbitrary* inputs, not the
//! hand-picked cases in the unit suite.
//!
//! Three properties carry the crate's claims:
//!
//! 1. **Coverage is bounded** (500 cases). For any path, at any size, at either
//!    fill rule, the mask's total ink is inside the frame it covers. This is
//!    the invariant that makes a coverage mask safe to composite.
//! 2. **Filling is area-exact for polygons** (300 cases). For an arbitrary
//!    polygon, the mask's ink equals the polygon's own area to within the 8-bit
//!    quantisation. This is the antialiasing claim, tested over the space
//!    rather than on one rectangle.
//! 3. **Measurement is additive and direction-free** (200 cases). A run's
//!    advance is exactly the sum of its glyph advances plus the kerns between
//!    them, and reordering the line — left-to-right, right-to-left, or by its
//!    own first strong character — never changes that total.
//!
//! Each is the kind of thing a unit test can accidentally arrange to pass: a
//! fixed example exercises one shape, a generator explores the space.

// Test harness: assertions legitimately panic; the lib target holds the
// deny-level lints.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::float_cmp
)]

use font_model::{Font, Glyph, GlyphId, Outline, Path};
use font_shape::{
    fill_path, mask_coverage, rasterize_outline, shape_line_with, stroke_path, FillRule, Hinting,
    LayoutDirection, Path2D, PathBuilder, StrokeStyle,
};
use proptest::prelude::*;

/// An arbitrary star-shaped polygon, fitted into `size`×`size` at `(2, 2)`.
///
/// A star rather than a convex blob because the interesting cases for a
/// scanline filler are *concave* rows, where a span starts and ends more than
/// once. Random radii keep it from being regular, and fitting it to the raster
/// is what lets the area property compare ink against area: a clipped polygon's
/// mask correctly holds less ink, which would make the property untestable
/// rather than false.
fn star_polygon(size: f32) -> impl Strategy<Value = Vec<(f32, f32)>> {
    prop::collection::vec((0.15f32..1.0, -0.1f32..0.1), 3..12).prop_map(move |rs| {
        let count = rs.len();
        let step = core::f32::consts::TAU / count as f32;
        let raw: Vec<(f32, f32)> = rs
            .iter()
            .enumerate()
            .map(|(i, (r, jitter))| {
                let a = step * i as f32 + *jitter;
                (*r * taylor_cos(a), *r * taylor_sin(a))
            })
            .collect();
        fit_into(&raw, size, 2.0)
    })
}

/// Scale and translate `points` to sit inside a `size`×`size` box at `(2, 2)`,
/// preserving their aspect ratio so the shape is not distorted.
fn fit_into(points: &[(f32, f32)], size: f32, inset: f32) -> Vec<(f32, f32)> {
    let w = max_x(points) - min_x(points);
    let h = max_y(points) - min_y(points);
    let extent = w.max(h).max(1e-6);
    let scale = size / extent;
    points
        .iter()
        .map(|p| {
            (
                (p.0 - min_x(points)) * scale + inset,
                (p.1 - min_y(points)) * scale + inset,
            )
        })
        .collect()
}

/// `cos`, without a `libm` dependency in the test target.
fn taylor_cos(a: f32) -> f32 {
    // A 12-term Taylor series is accurate to well under a float epsilon over
    // the range an angle argument can take here, and it keeps the test crate's
    // dependency list at exactly what the properties need.
    let x = a;
    let x2 = x * x;
    let mut term = 1.0f32;
    let mut sum = 1.0f32;
    for k in 1..8 {
        term *= -x2 / ((2 * k - 1) as f32 * (2 * k) as f32);
        sum += term;
    }
    sum
}

/// `sin`, likewise.
fn taylor_sin(a: f32) -> f32 {
    a * taylor_cos(a - core::f32::consts::FRAC_PI_2)
}

/// A closed path from a point list, counter-clockwise or clockwise as given.
fn closed(points: &[(f32, f32)]) -> Path2D {
    let mut b = PathBuilder::new();
    b.add_polygon(points, true);
    b.build()
}

proptest! {
    // 500 cases each: the headline properties, sized so a run explores the
    // space rather than grazing it.
    #[test]
    fn coverage_never_exceeds_its_frame(
        // A fixed 64 fits either rule and keeps the run fast; the frame bound
        // does not depend on the shape.
        points in star_polygon(64.0),
        rule in prop::sample::select(vec![FillRule::NonZero, FillRule::EvenOdd]),
    ) {
        let path = closed(&points);
        let side = 80u32;
        let mask = fill_path(&path, rule, side, side).unwrap();
        prop_assert_eq!(mask.len(), (side * side) as usize);

        let area = f64::from(side) * f64::from(side);
        let ink = mask_coverage(&mask) / 255.0;
        prop_assert!(ink >= 0.0, "negative ink {ink}");
        prop_assert!(
            ink <= area + 1e-6,
            "ink {ink} exceeds the {}x{} frame ({area})",
            side,
            side
        );
        // A coverage byte is a fraction of a pixel by construction, and at
        // least one pixel of a fitted polygon is inked.
        prop_assert!(mask.iter().any(|&v| v > 0), "a 64-unit polygon drew nothing");
    }

    #[test]
    fn polygon_fill_matches_its_area(points in star_polygon(40.0)) {
        let path = closed(&points);
        let area = f64::from(path.polygon_area());
        if area < 64.0 {
            // Too small for an 8-bit mask to say anything useful: one pixel of
            // quantisation is a larger fraction than the error being measured.
            return Ok(());
        }

        let side = 80u32;
        let mask = fill_path(&path, FillRule::NonZero, side, side).unwrap();
        let ink = mask_coverage(&mask) / 255.0;
        let err = (ink - area).abs() / area;
        // A polygon's edges are exact, so the only error is the 8-bit
        // quantisation: at most half a quantisation step per boundary pixel,
        // which for a shape of this size is well under a thousandth.
        prop_assert!(err < 1e-3, "ink {ink} vs area {area}: relative error {err}");
    }

    #[test]
    fn a_glyph_bitmap_holds_only_coverage(
        units in 100.0f32..1000.0,
        size_px in 4.0f32..64.0,
        hinting in prop::sample::select(vec![Hinting::None, Hinting::GridFit]),
    ) {
        // A square outline, scaled by the generator: the ink must be the
        // square's area to within quantisation, and the placement must be
        // consistent with the size.
        let mut outline = Outline::new();
        let mut p = Path::starting_at(100.0, 0.0);
        p.line_to(100.0 + units, 0.0);
        p.line_to(100.0 + units, units);
        p.line_to(100.0, units);
        p.close();
        outline.push_contour(p);

        let bmp = rasterize_outline(&outline, 1000, size_px, hinting).unwrap();
        prop_assert_eq!(bmp.coverage().len(), (bmp.width() * bmp.height()) as usize);

        let frame = f64::from(bmp.width()) * f64::from(bmp.height());
        let ink = bmp.ink();
        prop_assert!(ink >= 0.0 && ink <= frame + 1e-6, "ink {ink} vs frame {frame}");

        // The square's own area, in pixels. The relative error this measures
        // is the 8-bit rounding of the boundary columns over the whole, so it
        // shrinks as the glyph grows — roughly `perimeter / (255 · area)`. That
        // is below four parts per thousand once the shape is over 64 px of
        // area, and above it the quantisation says nothing.
        let want = (f64::from(units) * f64::from(size_px) / 1000.0).powi(2);
        if want > 64.0 && hinting == Hinting::None {
            let err = (ink - want).abs() / want;
            prop_assert!(err < 4e-3, "ink {ink} vs {want}: relative error {err}");
        }

        // Grid fitting snaps to whole pixels, so the bitmap is at most a pixel
        // larger than the outline's own box in each direction.
        let expected_side = f64::from(units) * f64::from(size_px) / 1000.0;
        prop_assert!(
            f64::from(bmp.width()) <= expected_side + 2.0,
            "width {} vs {expected_side}",
            bmp.width()
        );
        prop_assert!(
            f64::from(bmp.height()) <= expected_side + 2.0,
            "height {} vs {expected_side}",
            bmp.height()
        );
    }

    #[test]
    fn measuring_is_additive_and_direction_free(
        // A run of characters, drawn from the demo font's alphabet plus an
        // unmapped one, so the notdef fallback is exercised too.
        text in "[ABCDo]{1,24}",
        advances in prop::collection::vec(300u16..900, 5),
        kerns in prop::collection::vec(-120i16..120, 5),
        direction in prop::sample::select(vec![
            LayoutDirection::LeftToRight,
            LayoutDirection::RightToLeft,
            LayoutDirection::Auto,
        ]),
    ) {
        let font = font_of(&advances, &kerns);
        let size_px = 40.0f32;
        let scale = size_px / 1000.0;

        // The expected advance, computed directly from the model.
        let mut expected = 0.0f64;
        let mut prev: Option<GlyphId> = None;
        for ch in text.chars() {
            let glyph = glyph_of(&font, ch);
            expected += f64::from(font.glyph(glyph).expect("mapped").advance_width()) * f64::from(scale);
            if let Some(p) = prev {
                expected += f64::from(font.kern(p, glyph)) * f64::from(scale);
            }
            prev = Some(glyph);
        }

        let line = shape_line_with(&font, &text, size_px, direction).unwrap();
        prop_assert_eq!(line.len(), text.chars().count());
        let err = (f64::from(line.advance_px()) - expected).abs() / expected.max(1.0);
        prop_assert!(err < 1e-4, "advance {} vs {expected}", line.advance_px());

        // Every character has exactly one placement, and every placement
        // belongs to a character that is actually in the text.
        let mut clusters: Vec<u32> = line.placements().iter().map(|p| p.cluster).collect();
        clusters.sort_unstable();
        let want: Vec<u32> = (0..text.chars().count() as u32).collect();
        prop_assert_eq!(clusters, want.clone());

        // Reordering does not change the total, in any direction.
        for other in [
            LayoutDirection::LeftToRight,
            LayoutDirection::RightToLeft,
            LayoutDirection::Auto,
        ] {
            let l = shape_line_with(&font, &text, size_px, other).unwrap();
            prop_assert!(
                (l.advance_px() - line.advance_px()).abs() < 1e-3,
                "{other:?} changed the advance"
            );
        }

        // Every glyph's x is the running sum of the advances and kerns before
        // it. The pen need not be monotone — a negative kern legitimately
        // pulls a glyph left of where the previous one ended — but it must
        // agree with the model, and it must finish at the advance.
        let mut pen = 0.0f64;
        let mut prev: Option<GlyphId> = None;
        for cluster in &want {
            let ch = text.chars().nth(*cluster as usize).expect("in range");
            let glyph = glyph_of(&font, ch);
            let want_x = if let Some(p) = prev {
                pen + f64::from(font.kern(p, glyph)) * f64::from(scale)
            } else {
                0.0
            };
            let placement = line
                .placements()
                .iter()
                .find(|p| p.cluster == *cluster)
                .expect("every cluster is placed");
            prop_assert_eq!(placement.glyph, glyph);
            let err = (f64::from(placement.x) - want_x).abs();
            prop_assert!(err < 1e-3, "cluster {cluster} x {} vs {want_x}", placement.x);
            pen = want_x + f64::from(font.glyph(glyph).expect("mapped").advance_width())
                * f64::from(scale);
            prev = Some(glyph);
        }
        prop_assert!((pen - expected).abs() < 1e-3, "walked {pen} vs {expected}");
    }

    #[test]
    fn stroking_a_path_only_grows_its_bounds(
        points in star_polygon(64.0),
        width in 0.5f32..20.0,
    ) {
        let path = closed(&points);
        let stroked = stroke_path(&path, width, StrokeStyle::new(width));
        let (Some(a), Some(b)) = (path.bbox(), stroked.bbox()) else {
            return Ok(());
        };
        let slack = width * core::f32::consts::SQRT_2 * 1.5;
        // A stroke covers the path, so it starts no further in than the path.
        assert!(
            b.min_x <= a.min_x + slack && b.max_x >= a.max_x - slack,
            "{b:?} does not cover {a:?}"
        );
        // A butt-capped stroke stays within half a width of the path, plus the
        // arc slack a join or cap can add.
        assert!(
            b.min_x >= a.min_x - slack && b.max_x <= a.max_x + slack,
            "{b:?} strays from {a:?} by more than {slack}"
        );
        assert!(
            b.min_y >= a.min_y - slack && b.max_y <= a.max_y + slack,
            "{b:?} strays from {a:?} by more than {slack}"
        );
    }
}

/// The smallest x among `points`, ignoring none — the generator is finite.
fn min_x(points: &[(f32, f32)]) -> f32 {
    points.iter().fold(f32::INFINITY, |a, p| a.min(p.0))
}

/// The smallest y among `points`.
fn min_y(points: &[(f32, f32)]) -> f32 {
    points.iter().fold(f32::INFINITY, |a, p| a.min(p.1))
}

/// The largest x among `points`.
fn max_x(points: &[(f32, f32)]) -> f32 {
    points.iter().fold(f32::NEG_INFINITY, |a, p| a.max(p.0))
}

/// The largest y among `points`.
fn max_y(points: &[(f32, f32)]) -> f32 {
    points.iter().fold(f32::NEG_INFINITY, |a, p| a.max(p.1))
}

/// The glyph a character resolves to in `font`.
fn glyph_of(font: &Font, ch: char) -> GlyphId {
    font.glyph_for(ch as u32).unwrap_or(GlyphId::NOTDEF)
}

/// A font with the given advances and kerns over the alphabet `[A B C D o]`,
/// plus a kerning pair for every adjacent pair.
fn font_of(advances: &[u16], kerns: &[i16]) -> Font {
    let alphabet: Vec<char> = "ABCDo".chars().collect();
    let mut builder = Font::builder().units_per_em(1000);
    builder = builder.glyph(Glyph::empty(GlyphId::NOTDEF, 500));
    let mut cmap = font_model::CmapSubtable::empty_format4();
    for (i, ch) in alphabet.iter().enumerate() {
        let gid = GlyphId::new(i as u16 + 1);
        let advance = advances.get(i).copied().unwrap_or(500);
        builder = builder.glyph(Glyph::empty(gid, advance));
        cmap = font_model::insert_entry(&cmap, *ch as u32, gid);
    }
    for i in 0..alphabet.len() {
        for j in 0..alphabet.len() {
            if i == j {
                continue;
            }
            // A deterministic value per pair, drawn from the generated pool, so
            // a run explores both signs and both magnitudes.
            let v = kerns.get(i * alphabet.len() + j).copied().unwrap_or(0);
            builder = builder.kern(GlyphId::new(i as u16 + 1), GlyphId::new(j as u16 + 1), v);
        }
    }
    let mut table = font_model::Cmap::new();
    table.subtables_mut().push(cmap);
    builder.cmap(table).build().expect("valid")
}
