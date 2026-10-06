# font-shape

Glyph rasterization and text measurement — scanline fill, stroke expansion,
advance/kerning layout, and bidirectional reordering.

**Layer:** L2 — domain · **Estate deps:** `font-model` (L1) · **Runtime deps:**
`font-model`, `libm`, `thiserror` · `#![no_std]` + `alloc`.

## What this layer is

[`font-model`] owns a typeface: its outlines in font units, its `cmap`, its
metrics. That is the right shape for *holding* a font. It is the wrong shape
for *using* one — a screen needs pixels, and a text layout engine needs to know
where each glyph goes and how wide the line is.

`font-shape` is that half. It has **no rasterizer dependency at all**: the
scanline filler is the crate.

```text
   Font  ──shape_line──►  GlyphPlacement[]  ──►  pen positions
     │                       ▲
     │                       │ bidi levels, kerning, advances
     └──glyph_path──► Path2D ──fill_path──► coverage mask ──► pixels
                         ▲
              stroke_path ┘
```

## What's here

| Area | Entry points |
|---|---|
| transforms | [`Affine2D`] — 2×3 affine, application-order composition, inverse |
| paths | [`Path2D`], [`PathBuilder`] with a transform stack; curves stay curves |
| filling | [`fill_path`] — exact analytic coverage, [`FillRule`] |
| stroking | [`stroke_path`], [`StrokeStyle`], [`LineCap`], [`LineJoin`] |
| glyphs | [`rasterize_glyph`], [`rasterize_outline`], [`glyph_path`], [`Hinting`], [`GlyphBitmap`] |
| text | [`shape_line`], [`measure_run`], [`measure_glyph`], [`resolve_bidi`], [`LayoutDirection`] |

## The rasterizer

[`fill_path`] computes the **exact area** each pixel covers, integrating each
span's coverage analytically rather than supersampling. Two properties follow,
and both are asserted in the test suite:

- **An axis-aligned rectangle is exact.** On integer boundaries its interior is
  `255` and its exterior `0`; shifted half a pixel, each edge column is *exactly*
  `128`.
- **A mask's coverage sums to the polygon's area.** A 64-unit right triangle's
  ink matches `½·64·64` to within 0.1 %, and a 40-unit circle's matches `π·1600`
  to within 1 % — the residual being curve flattening, not the integration. That
  one number bounds every antialiasing claim this crate makes.

Device space is **y down**; pixel `(x, y)` covers `[x, x+1) × [y, y+1)`.
[`rasterize_glyph`] flips a font's y-up outline into that space and carries all
intermediate arithmetic in **26.6 fixed point** — 26 integer bits, 6 fractional,
one pixel exactly `64` — so a glyph lands on the same pixel grid at 12 px and at
144 px.

Stroking is expansion, not a distance function: a quad per segment, a wedge or a
fan per join, a quad or a fan per cap, every contour wound the same way, filled
by the same filler. Overlapping self-intersections are then correct under the
non-zero rule for free.

## Example

```rust
use font_model::{Font, Glyph, GlyphId, Outline, Path};
use font_shape::{rasterize_glyph, shape_line, Hinting};

// A 1000-upem font with one square glyph, 500 units wide.
let mut outline = Outline::new();
let mut p = Path::starting_at(50.0, 0.0);
p.line_to(450.0, 0.0);
p.line_to(450.0, 700.0);
p.line_to(50.0, 700.0);
p.close();
outline.push_contour(p);

let font = Font::builder()
    .units_per_em(1000)
    .glyph(Glyph::empty(GlyphId::NOTDEF, 500))
    .glyph(Glyph::new(GlyphId::new(1), 500, 50, outline))
    .build()
    .expect("valid");

// Rasterize at 20 px: an 8x14 block, 14 rows above the baseline.
let bmp = rasterize_glyph(&font, GlyphId::new(1), 20.0, Hinting::None)?;
assert_eq!((bmp.width(), bmp.height()), (8, 14));
assert_eq!(bmp.top(), 14);

// Measure the run: 500 units at 20 px/em = 10 px.
let line = shape_line(&font, "A", 20.0)?;
assert!((line.advance_px() - 10.0).abs() < 1e-4);
# Ok::<(), font_shape::ShapeError>(())
```

## Guarantees

- **Coverage is exact, not approximated.** The properties above, asserted in
  `tests/properties.rs` over 300 generated polygons rather than one rectangle.
- **Totality under untrusted geometry.** A path can carry NaN coordinates, a
  million-point subpath, or a transform that collapses the plane. Every public
  entry point returns a value or a typed [`ShapeError`] — never a panic, never
  an out-of-bounds write, never an unbounded allocation. The crate denies
  `clippy::unwrap_used`/`expect_used`/`panic`/`indexing_slicing` and
  `#![forbid(unsafe_code)]`, and the `font_shape_fuzz` target drives arbitrary
  bytes through every one of them.
- **Two fill rules, honestly.** [`FillRule::NonZero`] is the default and what a
  glyph wants; [`FillRule::EvenOdd`] punches a hole in a nested contour
  regardless of its direction. A five-pointed star drawn as one
  self-intersecting contour is the case that separates them, and it is a test.
- **Bidirectional reordering is a real UAX #9 subset.** Rules P2/P3, W1–W7,
  N1/N2, I1/I2, L1 and L2 are implemented, with `bidi_class` covering the
  strong, numeric, mark, and neutral classes. Kerning is applied in **logical**
  order — a kern pair is defined by reading order, not paint order — and
  reordering never changes a line's total advance.
- **Curves are kept as curves.** `Path2D` holds quadratic and cubic segments
  until rasterisation, where the pixel grid makes the right tolerance known.
  Bounding boxes solve each curve's derivative, so they hold the curve rather
  than its control hull.

## `render` example

```sh
cargo run --example render
cargo run --example render -- --size 48 --text "Wg80" --grid-fit
cargo run --example render -- --font tests/fixtures/minimal.ttf --out preview.pgm --png preview.png
```

Prints the line's metrics and per-cluster placements, an ASCII coverage map of
the composed line, and the stroke demo's command count. `--out` writes a PGM and
`--png` a PNG (the example carries a small stored-block deflate writer, so the
crate itself needs no compressor).

## Not in scope

Glyph-instruction (hinting bytecode) interpretation — [`Hinting::GridFit`] is a
coarse integer-grid fit, not TrueType's interpreter — variable-font axis
application, colour glyph rendering, subpixel positioning, text justification
and line breaking, and OpenType `GSUB`/`GPOS` substitution, which ride through
[`font-model`] opaquely.

## Layer

L2 — domain. Its only estate-internal dependency is `font-model` (L1), which
sits on `font-parse` (L0), so the chain is legal.

[`font-model`]: https://docs.rs/font-model
[`Affine2D`]: https://docs.rs/font-shape/latest/font_shape/struct.Affine2D.html
[`Path2D`]: https://docs.rs/font-shape/latest/font_shape/struct.Path2D.html
[`PathBuilder`]: https://docs.rs/font-shape/latest/font_shape/struct.PathBuilder.html
[`fill_path`]: https://docs.rs/font-shape/latest/font_shape/fn.fill_path.html
[`FillRule`]: https://docs.rs/font-shape/latest/font_shape/enum.FillRule.html
[`stroke_path`]: https://docs.rs/font-shape/latest/font_shape/fn.stroke_path.html
[`StrokeStyle`]: https://docs.rs/font-shape/latest/font_shape/struct.StrokeStyle.html
[`LineCap`]: https://docs.rs/font-shape/latest/font_shape/enum.LineCap.html
[`LineJoin`]: https://docs.rs/font-shape/latest/font_shape/enum.LineJoin.html
[`rasterize_glyph`]: https://docs.rs/font-shape/latest/font_shape/fn.rasterize_glyph.html
[`rasterize_outline`]: https://docs.rs/font-shape/latest/font_shape/fn.rasterize_outline.html
[`glyph_path`]: https://docs.rs/font-shape/latest/font_shape/fn.glyph_path.html
[`Hinting`]: https://docs.rs/font-shape/latest/font_shape/enum.Hinting.html
[`GlyphBitmap`]: https://docs.rs/font-shape/latest/font_shape/struct.GlyphBitmap.html
[`shape_line`]: https://docs.rs/font-shape/latest/font_shape/fn.shape_line.html
[`measure_run`]: https://docs.rs/font-shape/latest/font_shape/fn.measure_run.html
[`measure_glyph`]: https://docs.rs/font-shape/latest/font_shape/fn.measure_glyph.html
[`resolve_bidi`]: https://docs.rs/font-shape/latest/font_shape/fn.resolve_bidi.html
[`LayoutDirection`]: https://docs.rs/font-shape/latest/font_shape/enum.LayoutDirection.html
[`ShapeError`]: https://docs.rs/font-shape/latest/font_shape/enum.ShapeError.html
[`FillRule::NonZero`]: https://docs.rs/font-shape/latest/font_shape/enum.FillRule.html#variant.NonZero
[`FillRule::EvenOdd`]: https://docs.rs/font-shape/latest/font_shape/enum.FillRule.html#variant.EvenOdd