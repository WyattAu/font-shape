//! Regression: hostile-extent geometry must return a mask, never spin.
//!
//! Found by `font-conformance` (case `huge_extent_polygon`, 2026-10-17):
//! a ±1e30 triangle saturated `add_band`'s column walk at the `i32` bounds
//! and `fill_path` never returned — a silent hang, which violates the crate
//! discipline ("a path can carry hostile geometry ... the rasterizer must
//! still return a mask"). The fix clamps the walk to the mask width.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    // Justification: test target lifts the lib-target denials at file scope
    // per the estate decoder-discipline pattern; assertions need them.
    clippy::panic
)]

use font_shape::{fill_path, stroke_path, FillRule, PathBuilder, StrokeStyle};

#[test]
fn hostile_extent_triangle_fills() {
    let mut b = PathBuilder::new();
    b.add_polygon(&[(1e30, 1e30), (-1e30, 0.0), (0.0, -1e30)], true);
    let path = b.build();

    let mask =
        fill_path(&path, FillRule::NonZero, 8, 8).expect("hostile extent fills, never hangs");
    assert_eq!(mask.len(), 64);
    assert!(
        mask.iter().all(|&v| v <= 255),
        "coverage stays a u8 coverage value"
    );
}

#[test]
fn stroked_hostile_extent_triangle_fills() {
    let mut b = PathBuilder::new();
    b.add_polygon(&[(1e30, 1e30), (-1e30, 0.0), (0.0, -1e30)], true);
    let path = b.build();

    let stroked = stroke_path(&path, 1.0, StrokeStyle::new(1.0));
    let mask = fill_path(&stroked, FillRule::NonZero, 8, 8)
        .expect("stroked hostile extent fills, never hangs");
    assert_eq!(mask.len(), 64);
}

#[test]
fn hostile_extent_and_nan_mixed_still_total() {
    // The clamp must hold when the extremes reach the walk from *several*
    // rows' worth of edges, not just one band.
    let mut b = PathBuilder::new();
    b.add_polygon(
        &[(1e30, 1e30), (-1e30, 1e29), (0.0, -1e30), (-1e29, 0.0)],
        true,
    );
    let path = b.build();
    let mask = fill_path(&path, FillRule::EvenOdd, 16, 16).expect("fills");
    assert_eq!(mask.len(), 256);
}
