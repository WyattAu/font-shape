# Changelog

All notable changes to this project are documented in this file. Format:
[Keep a Changelog](https://keepachangelog.com/); versions follow
[semver](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-10-06

### Added

- **`Affine2D`** — a 2×3 affine transform in the order a font's composite glyph
  transforms use. `then` composes in *application* order (`a.then(&b)` applies
  `a` and then `b`), which is the nesting a `push_transform`/`pop_transform`
  stack wants; `inverse` reports `SingularAffine` rather than returning a matrix
  of infinities; `scale_factor` is the cube root of the absolute determinant,
  so a reflection and its mirror agree.
- **`Path2D` / `PathBuilder`** — device-space paths of absolute line, quad, and
  cubic segments. Curves stay curves until rasterisation, where the pixel grid
  makes the right tolerance known. `bbox` solves each curve's derivative for
  its axis extrema, so the box holds the curve rather than its control hull.
  The builder applies the transform stack **at add time**, and the stack's
  newest transform is applied first, so a `translate(10)` outside a
  `scale(2)` puts a unit square at (10, 0) rather than (20, 0). An unbalanced
  `pop_transform` is `UnbalancedPop` rather than a silent identity.
  `flatten` drops a degenerate subpath and reports each subpath's real
  closedness, which is what lets the stroker put a cap where a join does not
  belong.
- **`fill_path`** — the scanline rasteriser, and the reason this crate exists.
  It computes the **exact analytic area** each pixel covers: within a band where
  the edge set is fixed, the span's overlap with a column is piecewise linear in
  `y`, split at every edge-crossing and summed as trapezoids. An axis-aligned
  rectangle on integer boundaries is exactly `255`/`0`; shifted half a pixel,
  exactly `128` per edge column; a 64-unit triangle's ink matches `½·64·64` to
  within 0.1 % and a 40-unit circle's matches `π·1600` to within 1 %. A row is
  split at every active edge's endpoints, which is what makes curved rows come
  out right. Both fill rules: `NonZero` (the default, and what a glyph wants)
  and `EvenOdd`. `mask_coverage` sums a mask, `path_pixel_bounds` reports a
  path's tight pixel bounds, and a raster over `MAX_MASK_ELEMENTS` is
  `RasterTooLarge` rather than an allocation.
- **`stroke_path`** — stroke expansion: a quad per segment, a miter wedge or a
  round fan or a bevel triangle per join, a quad or a half-disc fan per cap,
  every contour emitted in the same rotational direction so filling the union
  under the non-zero rule *is* a union. Round arcs are `ARC_STEPS`-gons,
  `DEFAULT_MITER_LIMIT` matches SVG, and a zero, negative, or non-finite width
  has no region and yields an empty path. `Path2D::reverse` flips a path's
  winding while keeping its geometry, including for a lone `MoveTo`.
- **`rasterize_glyph` / `rasterize_outline` / `glyph_path` / `GlyphBitmap`** —
  glyph rasterisation in **26.6 fixed point**, `size_px · 64 / units_per_em`,
  with the y flip from font space into the rasteriser's y-down device space
  folded into the scale. `GlyphBitmap` carries 8-bit coverage plus the
  `left`/`top` placement a caller needs to put it back in a run, and a
  `coverage_at`/`row`/`ink_bounds` view that is bounds-checked. An outline with
  no contours rasterizes to an empty bitmap rather than to an error; a glyph
  whose box exceeds the mask cap is refused before allocating.
- **`Hinting`** — `None` (the default, unhinted) and `GridFit`, which snaps
  every outline point to the pixel grid and rounds bearings. A coarse
  integer-grid fit, not TrueType's interpreter, and documented as such.
- **`shape_line` / `measure_run` / `measure_glyph` / `ShapedLine` /
  `GlyphPlacement`** — advances, kerning, and layout. Kerning is applied to
  every adjacent pair in **logical** order, because a kern pair is defined by
  reading order and not by the order glyphs happen to be painted. Each
  `GlyphPlacement` carries its glyph, its cluster index, its resolved embedding
  level, its pen position, and its advance; `ShapedLine::by_cluster` inverts the
  visual order back to reading order for a selection or extraction path. An
  unmapped character falls back to `.notdef` so a run measures at the width a
  renderer would actually draw.
- **`resolve_bidi` / `BidiLevels` / `LayoutDirection`** — a real subset of UAX
  #9: rules P2/P3 for the paragraph level, W1–W7 for the class resolution, N1
  and N2 for neutrals, I1 and I2 for the implicit levels, L1 for trailing
  whitespace, and L2 for the visual reorder. `bidi_class` distinguishes `L`,
  `R`, `Al`, `En`, `An`, `Nsm`, `Et`, and neutrals. Numbers inside a
  right-to-left run take level 2 so their digits keep their own order, and
  `set_explicit_levels` is how a caller injects direction for context this
  function cannot see. Reordering never changes a line's total advance.
- **`scaled_metrics` / `DeviceMetrics`** — a font's metrics resolved to device
  pixels, with `descent` reported as the positive number of pixels below the
  baseline that it actually is.
- **`error`** — `ShapeError`, exhaustive and not `#[non_exhaustive]`, so a new
  variant is a semver-minor event by design: `SingularAffine`, `UnbalancedPop`,
  `InvalidSize`, `InvalidScale`, `UnknownGlyph`, `NonFiniteCoordinate`,
  `RasterTooLarge`, `DegenerateUnitsPerEm`.
- **`render` example** — builds a demo font, rasterizes a line, and prints the
  metrics, the per-cluster placements, an ASCII coverage map, and the stroke
  demo; `--out` writes a PGM and `--png` a PNG through a small stored-block
  deflate writer, so the crate itself needs no compressor. `--font` runs it
  against a real TrueType file through `font-model`'s parser.
- **Tests** — 130 unit tests, 14 integration tests driving the crate the way a
  renderer does, 5 `proptest` properties (coverage bounded by its frame, a
  polygon's fill matching its area, a glyph bitmap holding only coverage,
  measurement additive and direction-free, a stroke staying near what it
  strokes), 30 doctests, and a `font_shape_fuzz` target asserting totality plus
  those same invariants over arbitrary bytes.

[Unreleased]: https://github.com/WyattAu/font-shape/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/WyattAu/font-shape/releases/tag/v0.1.0