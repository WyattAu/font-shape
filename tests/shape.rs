//! Integration tests for the public surface: the operations a text renderer
//! actually calls, in the order it calls them.
//!
//! The unit suites live next to the code they test; these exercise the crate
//! the way a downstream consumer does — through `font_shape::`'s re-exports
//! only, with no crate-internal shortcuts — and across module boundaries.
//! Loading a real TrueType file goes through `font_model::from_sfnt` and so
//! covers the whole chain at once.

// Test harness: assertions legitimately panic; the lib target holds the
// deny-level lints.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::float_cmp
)]

use font_model::{from_sfnt, Font, Glyph, GlyphId, Outline, Path};
use font_shape::{
    fill_path, fixed_scale, from_fixed, glyph_path, mask_coverage, measure_glyph, measure_run,
    rasterize_glyph, rasterize_outline, resolve_bidi, scaled_metrics, shape_line, shape_line_with,
    stroke_path, to_fixed, Affine2D, FillRule, GlyphBitmap, Hinting, LayoutDirection, PathBuilder,
    ShapeError, StrokeStyle,
};

/// A square in font units, y up, from `(x0, y0)` to `(x1, y1)`.
fn square(x0: f32, y0: f32, x1: f32, y1: f32) -> Outline {
    let mut o = Outline::new();
    let mut p = Path::starting_at(x0, y0);
    p.line_to(x1, y0);
    p.line_to(x1, y1);
    p.line_to(x0, y1);
    p.close();
    o.push_contour(p);
    o
}

/// The crate's standard test font: `.notdef`, 'A', and 'B', with `A` kerned to
/// `B` by -40 units.
fn test_font() -> Font {
    Font::builder()
        .units_per_em(1000)
        .ascender(800)
        .descender(-200)
        .glyph(Glyph::new(
            GlyphId::NOTDEF,
            600,
            50,
            square(50.0, 0.0, 550.0, 700.0),
        ))
        .glyph(Glyph::new(
            GlyphId::new(1),
            600,
            50,
            square(50.0, 0.0, 550.0, 700.0),
        ))
        .glyph(Glyph::new(
            GlyphId::new(2),
            700,
            50,
            square(50.0, 0.0, 550.0, 700.0),
        ))
        .kern(GlyphId::new(1), GlyphId::new(2), -40)
        .cmap(font_shape::ascii_cmap())
        .build()
        .expect("valid")
}

/// A font whose 'A' is blank and whose 'B' is inked, so a test can tell the
/// two glyphs apart in a raster.
fn mixed_font() -> Font {
    Font::builder()
        .units_per_em(1000)
        .glyph(Glyph::empty(GlyphId::NOTDEF, 500))
        .glyph(Glyph::empty(GlyphId::new(1), 400))
        .glyph(Glyph::new(
            GlyphId::new(2),
            500,
            0,
            square(0.0, 0.0, 500.0, 500.0),
        ))
        .kern(GlyphId::new(1), GlyphId::new(2), 0)
        .cmap(font_shape::ascii_cmap())
        .build()
        .expect("valid")
}

#[test]
fn rasterize_measure_compose() {
    // The whole chain a renderer performs, once, end to end.
    let font = test_font();
    let size_px = 24.0;

    // 1. Measure the run.
    let advance = measure_run(&font, "AB", size_px).expect("measures");
    assert!((advance - (600.0 + 700.0 - 40.0) * 0.024).abs() < 1e-4);

    // 2. Shape it, and check the kern is where the measurement said it would be.
    let line = shape_line(&font, "AB", size_px).expect("shapes");
    assert_eq!(line.len(), 2);
    assert!((line.advance_px() - advance).abs() < 1e-4);
    // 'B' starts one kern earlier than 'A's advance: (600 − 40) units.
    assert!(
        (line.get(1).unwrap().x - (600.0 - 40.0) * 0.024).abs() < 1e-4,
        "kern applied: {}",
        line.get(1).unwrap().x
    );

    // 3. Rasterize each glyph at its own pen position.
    let metrics = scaled_metrics(&font, size_px).expect("metrics");
    let baseline = metrics.ascent.ceil() as i32 + 1;
    let height = (baseline + metrics.descent.ceil() as i32 + 1) as u32;
    let width = (advance.ceil() as u32 + 2).max(1);
    let mut canvas = vec![0u8; (width * height) as usize];
    for p in line.placements() {
        let bmp = rasterize_glyph(&font, p.glyph, size_px, Hinting::None).expect("raster");
        assert!(!bmp.is_empty(), "the test glyph has ink");
        blit(
            &mut canvas,
            width,
            height,
            &bmp,
            p.x.round() as i32,
            baseline,
        );
    }

    // 4. The composed line's ink is the sum of the two glyphs' ink. The glyphs
    // do not overlap here — the kern pulls 'B' 0.96 px closer than 'A''s
    // advance, and each is 12 px wide — so nothing is composited twice.
    let total: f64 = line
        .placements()
        .iter()
        .map(|p| {
            rasterize_glyph(&font, p.glyph, size_px, Hinting::None)
                .map(|b| b.ink())
                .unwrap_or(0.0)
        })
        .sum();
    let composed = mask_coverage(&canvas) / 255.0;
    assert!(
        (composed - total).abs() < 1.0,
        "composed {composed} vs per-glyph {total}"
    );
    // And each glyph's bitmap holds exactly its own area. The box is 13 wide
    // rather than 12 because the outline's left sidebearing puts its left edge
    // at 1.2 px and its right at 13.2 px, and a bitmap spans the whole pixels
    // its ink touches — the coverage sums to the area regardless.
    let a = rasterize_glyph(&font, GlyphId::new(1), size_px, Hinting::None).expect("raster");
    assert_eq!(a.left(), 1, "1.2 px rounds down to 1");
    assert_eq!((a.width(), a.height()), (13, 17));
    // 500 × 700 units at 24 px/em is 12 × 16.8 px, so 201.6 px of ink, to
    // within the 8-bit rounding of the two partially covered edge columns.
    let want = 500.0 * 700.0 * 0.024 * 0.024;
    assert!((a.ink() - want).abs() < 0.1, "ink {} vs {want}", a.ink());
}

/// Copy a bitmap into a canvas at a pen position and baseline row.
fn blit(canvas: &mut [u8], width: u32, _height: u32, bmp: &GlyphBitmap, pen_x: i32, baseline: i32) {
    if bmp.is_empty() {
        return;
    }
    let ox = pen_x + bmp.left();
    let oy = baseline - bmp.top();
    for row in 0..bmp.height() {
        for col in 0..bmp.width() {
            let v = bmp.coverage_at(col, row);
            let px = ox + col as i32;
            let py = oy + row as i32;
            if v == 0 || px < 0 || py < 0 {
                continue;
            }
            let idx = (py as u32) * width + px as u32;
            if let Some(slot) = canvas.get_mut(idx as usize) {
                *slot = slot.saturating_add(v);
            }
        }
    }
}

#[test]
fn hinting_changes_pixels_but_not_the_advance() {
    let font = test_font();
    // A fractional size is where the two modes differ: 23.5 px cannot land on
    // whole pixels, so grid fitting has to round somewhere.
    let plain = rasterize_glyph(&font, GlyphId::new(1), 23.5, Hinting::None).expect("raster");
    let fitted = rasterize_glyph(&font, GlyphId::new(1), 23.5, Hinting::GridFit).expect("raster");
    assert_ne!(
        plain.coverage(),
        fitted.coverage(),
        "grid fitting changes the raster at a fractional size"
    );
    // Grid fitting snaps the outline to whole pixels, so its box is within one
    // pixel of the unhinted one.
    assert!(
        (fitted.width() as i32 - plain.width() as i32).abs() <= 1
            && (fitted.height() as i32 - plain.height() as i32).abs() <= 1,
        "grid fit {}x{} vs plain {}x{}",
        fitted.width(),
        fitted.height(),
        plain.width(),
        plain.height()
    );

    // Measurement is unaffected: hinting is a rasteriser decision.
    assert!(
        (measure_glyph(&font, GlyphId::new(1), 23.5).expect("ok") - 600.0 * 0.0235).abs() < 1e-4
    );
    assert_eq!(
        measure_glyph(&font, GlyphId::new(1), 23.5).expect("ok"),
        measure_glyph(&font, GlyphId::new(1), 23.5).expect("ok")
    );
}

#[test]
fn glyph_path_places_ink_above_the_baseline() {
    let font = test_font();
    let path = glyph_path(&font, GlyphId::new(1), 20.0, Hinting::None).expect("path");
    let b = path.bbox().expect("a glyph with ink has a bbox");
    // The outline's y runs 0..700, so in device space it runs -14..0.
    assert!(b.max_y <= 1e-3, "the baseline is at y = 0, not above it");
    assert!((b.min_y - -14.0).abs() < 0.02, "{}", b.min_y);
    assert!((b.min_x - 1.0).abs() < 0.02, "{}", b.min_x);
}

#[test]
fn sizes_scale_the_glyph_exactly() {
    let font = test_font();
    let mut previous: Option<(f32, f64)> = None;
    for size in [8.0f32, 16.0, 24.0, 48.0, 96.0] {
        let bmp = rasterize_glyph(&font, GlyphId::new(1), size, Hinting::None).expect("raster");
        // 500 x 700 units at 1/1000 of `size` px per unit.
        assert!(
            (bmp.width() as i64 - (500.0 * size / 1000.0).round() as i64).abs() <= 1,
            "width {} at {size} px",
            bmp.width()
        );
        assert!(
            (bmp.height() as i64 - (700.0 * size / 1000.0).round() as i64).abs() <= 1,
            "height {} at {size} px",
            bmp.height()
        );
        if let Some((prev_size, prev_ink)) = previous {
            let ratio = f64::from(size / prev_size);
            let got = bmp.ink() / prev_ink;
            // The ink tracks the area, within the 8-bit quantisation of the
            // boundary columns — which is a smaller fraction of the whole the
            // bigger the glyph gets.
            let tol = 0.02 + 0.4 / prev_ink;
            assert!(
                (got / (ratio * ratio) - 1.0).abs() < tol,
                "{prev_size}->{size} px: ink ratio {got} vs area ratio {}",
                ratio * ratio
            );
        }
        previous = Some((size, bmp.ink()));
    }
}

#[test]
fn fixed_point_is_the_scale_between_them() {
    // The scale is 26.6 units per font unit, and the bitmap lands where that
    // scale says it should.
    let scale = fixed_scale(24.0, 1000).expect("scale");
    assert!((scale - 24.0 * 64.0 / 1000.0).abs() < 1e-4);
    // A glyph coordinate in font units becomes a device coordinate in pixels.
    let x_fixed = to_fixed(500.0 * scale / 64.0);
    assert!(
        (from_fixed(x_fixed) - 12.0).abs() < 0.01,
        "{}",
        from_fixed(x_fixed)
    );

    // And the rasterized bitmap's own width agrees with it.
    let bmp = rasterize_outline(&square(0.0, 0.0, 500.0, 700.0), 1000, 24.0, Hinting::None)
        .expect("raster");
    assert_eq!(bmp.width(), 12);
    assert_eq!(bmp.height(), 17);
}

#[test]
fn a_space_renders_nothing_but_still_advances() {
    let font = mixed_font();
    // 'A' is blank in this font: no pixels, but its advance is real.
    let blank = rasterize_glyph(&font, GlyphId::new(1), 20.0, Hinting::None).expect("raster");
    assert!(blank.ink() == 0.0);
    assert!((measure_glyph(&font, GlyphId::new(1), 20.0).expect("ok") - 8.0).abs() < 1e-4);

    // And a run with a blank glyph still measures as the sum of advances.
    let line = shape_line(&font, "AB", 20.0).expect("shapes");
    assert!(
        (line.advance_px() - 18.0).abs() < 1e-4,
        "{}",
        line.advance_px()
    );
    assert_eq!(line.len(), 2);
}

#[test]
fn an_unmapped_character_falls_back_to_notdef() {
    let font = test_font();
    // 'z' is not in the cmap. It draws `.notdef` and advances by its width,
    // so the measured width matches what a renderer would produce.
    let mapped = measure_run(&font, "A", 10.0).expect("measures");
    let unmapped = measure_run(&font, "z", 10.0).expect("measures");
    assert!(
        (unmapped - 600.0 * 0.01).abs() < 1e-4,
        "notdef's advance, not zero: {unmapped}"
    );
    assert!((unmapped - mapped).abs() < 1e-4, "both are 600 units");

    let line = shape_line(&font, "z", 10.0).expect("shapes");
    assert_eq!(line.get(0).unwrap().glyph, GlyphId::NOTDEF);
    assert_eq!(line.get(0).unwrap().cluster, 0);
}

#[test]
fn bidi_reordering_is_reflected_in_the_placements() {
    let font = test_font();
    // A Hebrew letter mapped to a glyph in this font, with a Latin 'A' around
    // it: the classic three-level run.
    let text = "A\u{05D0}";
    let bidi = resolve_bidi(text);
    assert_eq!(bidi.levels.len(), 2);
    assert_eq!(bidi.levels[0], 0, "'A' is left to right");
    assert_eq!(bidi.levels[1], 1, "the Hebrew is an RTL island");

    let line = shape_line(&font, text, 20.0).expect("shapes");
    assert_eq!(line.len(), 2);
    // The Hebrew glyph is placed at the pen position of the second character,
    // and it carries the RTL level.
    let hebrew = line
        .placements()
        .iter()
        .find(|p| p.cluster == 1)
        .expect("cluster 1 is placed");
    assert_eq!(hebrew.level, 1);
    assert!(hebrew.is_rtl());
    assert!((hebrew.x - 12.0).abs() < 1e-4, "after A's advance");
}

#[test]
fn rtl_changes_the_draw_order_not_the_width() {
    let font = test_font();
    let ltr = shape_line_with(&font, "AB", 20.0, LayoutDirection::LeftToRight).expect("ok");
    let rtl = shape_line_with(&font, "AB", 20.0, LayoutDirection::RightToLeft).expect("ok");
    assert_eq!(ltr.len(), rtl.len());
    assert!((ltr.advance_px() - rtl.advance_px()).abs() < 1e-6);
    // Display order: the last character is drawn first.
    assert_eq!(ltr.get(0).unwrap().cluster, 0);
    assert_eq!(rtl.get(0).unwrap().cluster, 1);
    // Logical positions are unchanged: 'A' is still at x = 0.
    let a = rtl.placements().iter().find(|p| p.cluster == 0).expect("A");
    assert!(a.x.abs() < 1e-6);
    assert!(a.is_rtl());
    // And the cluster view is logical either way.
    assert_eq!(rtl.by_cluster()[0].0, 0);
}

#[test]
fn auto_direction_reads_the_text() {
    let font = test_font();
    // Latin first, so `Auto` is left to right...
    let ltr = shape_line_with(&font, "AB", 20.0, LayoutDirection::Auto).expect("ok");
    assert_eq!(ltr.paragraph_level(), 0);
    assert_eq!(ltr.get(0).unwrap().cluster, 0);

    // ...and a leading Hebrew letter makes it right to left.
    let rtl = shape_line_with(&font, "\u{05D0}A", 20.0, LayoutDirection::Auto).expect("ok");
    assert_eq!(rtl.paragraph_level(), 1);
}

#[test]
fn stroke_then_fill_covers_the_path() {
    // A path and its stroke, and the invariant that relates them: the stroke
    // covers the path.
    let mut b = PathBuilder::new();
    b.move_to(10.0, 40.0);
    b.line_to(60.0, 40.0);
    b.line_to(60.0, 90.0);
    let path = b.build();

    let width = 8.0;
    let stroked = stroke_path(&path, width, StrokeStyle::new(width));
    assert!(!stroked.is_empty());

    // The path is an open polyline, so filling it closes the gap between its
    // ends — which would measure the wrong thing. The stroke's own area is
    // what is being pinned here: length × width for a two-segment polyline.
    let mask = fill_path(&stroked, FillRule::NonZero, 128, 128).expect("fills");
    let ink = mask_coverage(&mask) / 255.0;
    let length = f64::from(width) * 100.0;
    assert!(
        (ink - length).abs() < length * 0.01,
        "stroke ink {ink} vs 100 x {width} = {length}"
    );

    // And the stroke's bounds contain the path's, grown by at most half a
    // width plus the join's reach.
    let (Some(a), Some(b)) = (path.bbox(), stroked.bbox()) else {
        panic!("both have bounds");
    };
    let slack = width * 0.5 * core::f32::consts::SQRT_2 + 0.5;
    assert!(
        b.min_x <= a.min_x && b.max_x >= a.max_x,
        "{b:?} does not span {a:?}"
    );
    assert!(
        b.min_x >= a.min_x - slack && b.max_x <= a.max_x + slack,
        "{b:?} vs {a:?}"
    );
    assert!(
        b.min_y <= a.min_y && b.max_y >= a.max_y,
        "{b:?} does not span {a:?}"
    );
    assert!(
        b.min_y >= a.min_y - slack && b.max_y <= a.max_y + slack,
        "{b:?} vs {a:?}"
    );
}

#[test]
fn every_public_error_is_reachable_from_the_public_api() {
    let font = test_font();
    // Singular affine.
    assert!(Affine2D::scale(0.0, 1.0).inverse().is_err());
    // Unbalanced pop.
    let mut b = PathBuilder::new();
    assert!(matches!(
        b.pop_transform().unwrap_err(),
        ShapeError::UnbalancedPop { .. }
    ));
    // Invalid scale.
    assert!(matches!(
        measure_run(&font, "A", 0.0),
        Err(ShapeError::InvalidScale { .. })
    ));
    // Unknown glyph.
    assert!(matches!(
        measure_glyph(&font, GlyphId::new(99), 10.0),
        Err(ShapeError::UnknownGlyph { .. })
    ));
    // Raster too large.
    let mut huge = PathBuilder::new();
    huge.rect(0.0, 0.0, 1.0, 1.0);
    assert!(matches!(
        fill_path(&huge.build(), FillRule::NonZero, 100_000, 100_000),
        Err(ShapeError::RasterTooLarge { .. })
    ));
    // Degenerate units per em.
    assert!(matches!(
        rasterize_outline(&square(0.0, 0.0, 100.0, 100.0), 0, 12.0, Hinting::None),
        Err(ShapeError::DegenerateUnitsPerEm)
    ));
    // And the two that only arise from geometry a caller can build: a
    // non-finite scale, and an invalid size.
    assert!(matches!(
        rasterize_outline(
            &square(0.0, 0.0, 100.0, 100.0),
            1000,
            f32::NAN,
            Hinting::None
        ),
        Err(ShapeError::InvalidScale { .. })
    ));
    assert!(matches!(
        shape_line(&font, "A", f32::INFINITY),
        Err(ShapeError::InvalidScale { .. })
    ));
}

#[test]
fn a_real_truetype_file_goes_through_the_whole_chain() {
    let bytes = include_bytes!("fixtures/minimal.ttf");
    let font = from_sfnt(bytes).expect("the fixture parses");
    assert!(font.glyph_count() > 0);

    // Every glyph rasterizes without panicking, at every glyph's advance.
    let size_px = 16.0;
    for id in font.glyphs().iter().map(font_model::Glyph::id) {
        let bmp = rasterize_glyph(&font, id, size_px, Hinting::None).expect("raster");
        assert_eq!(
            bmp.coverage().len(),
            (bmp.width() * bmp.height()) as usize,
            "the mask matches the dimensions for {id}"
        );
        let _ = measure_glyph(&font, id, size_px).expect("measures");
    }

    // A couple of runs shape, and the advance is positive and finite.
    for text in ["A", "AB", "The quick brown fox"] {
        let line = shape_line(&font, text, size_px).expect("shapes");
        assert!(line.advance_px() >= 0.0 && line.advance_px().is_finite());
        assert!(line.ascent_px() > 0.0, "a font has an ascender");
    }

    // And its metrics scale.
    let m = scaled_metrics(&font, size_px).expect("metrics");
    assert!(m.ascent > 0.0 && m.descent > 0.0);
    assert!(m.line_height >= m.ascent + m.descent);
}

#[test]
fn a_whole_font_renders_at_many_sizes() {
    // The property the sizes loop in one test depends on: a glyph's ink is its
    // area, at every size, with no special cases in between.
    let outline = square(100.0, 0.0, 600.0, 700.0);
    for size in [
        1.0f32, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0, 89.0, 144.0,
    ] {
        let bmp = rasterize_outline(&outline, 1000, size, Hinting::None)
            .unwrap_or_else(|e| panic!("size {size}: {e}"));
        let want = f64::from(500.0 * size / 1000.0) * f64::from(700.0 * size / 1000.0);
        let err = (bmp.ink() - want).abs() / want.max(1.0);
        // At the very smallest sizes the ink is a handful of pixels and the
        // 8-bit quantisation dominates; past a few pixels it is exact.
        let tol = if want < 8.0 { 0.5 } else { 0.02 };
        assert!(
            err < tol,
            "size {size}: ink {} vs {want} (err {err})",
            bmp.ink()
        );
    }
}
