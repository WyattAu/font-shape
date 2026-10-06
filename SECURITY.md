# Security Policy — font-shape

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅        |

## Reporting a vulnerability

Report privately via [GitHub security advisories] for this repository, or
email **wyatt_au@protonmail.com**. Do **not** open a public issue for
security reports.

You will receive an acknowledgement within **72 hours**. Coordinated
disclosure: we ask for up to 90 days before public disclosure while a
patch ships.

## Scope notes

`font-shape` is a pure computation library over caller-supplied geometry and
caller-supplied fonts. Security considerations for integrators:

- **No network, filesystem, or process state** in the library. The one
  filesystem touch is in the `render` example, which is not shipped in the
  library target.
- **Geometry is the untrusted-input edge.** A `Path2D` can arrive from a font,
  from a user drawing tool, or from a network protocol, and it can carry NaN
  and infinite coordinates, degenerate contours, and a million points in one
  subpath. Everything here is total: `fill_path`, `stroke_path`,
  `glyph_path`, `rasterize_outline`, `measure_run`, and `shape_line` each
  return a value or a typed `ShapeError` — never a panic, never an
  out-of-bounds write. The crate denies `clippy::unwrap_used`, `expect_used`,
  `panic`, and `indexing_slicing`, and `#![forbid(unsafe_code)]`.
- **Allocation is bounded by an explicit cap.** `fill_path` refuses a raster
  larger than `MAX_MASK_ELEMENTS` (4096 × 4096 = 16 MiB of coverage) with
  `RasterTooLarge` rather than attempting it, and a glyph whose own bounding
  box exceeds that cap is refused before any mask is allocated. A zero-sized
  raster is empty, not an error. The remaining allocations — the flattened
  path, the edge list, the stroke expansion — are proportional to the input the
  caller already holds, so a caller bounding a hostile path's point count
  bounds them too. The fuzz target caps point, contour, and raster sizes for
  the same reason.
- **Non-finite coordinates are dropped, not repaired.** A non-finite coordinate
  is skipped by the edge collector and by `polygon_area`, and a glyph outline
  whose bounding box is non-finite rasterizes to an empty bitmap rather than to
  a plausible-looking wrong picture. The 26.6 conversion maps NaN to zero and
  saturates at `i32::MAX`/`MIN`, so an extreme coordinate becomes a
  correspondingly extreme — and immediately refused — raster, never an
  integer wrap. If silent dropping is wrong for your use, check
  `font_model::Font::validate` first: a repaired coordinate would be a wrong
  outline that looks right.
- **`Hinting::GridFit` is not an interpreter.** It snaps outline points to the
  pixel grid and rounds bearings. It does not execute glyph instructions, so
  there is no bytecode interpreter to attack — and equally, it is not a
  defence against a font whose *outlines* are hostile, which the geometry rules
  above already cover.
- **Bidi resolution is a pure function of the text.** `resolve_bidi` and
  `shape_line` allocate in proportion to the character count and never index
  outside it; `set_explicit_levels` clamps a caller-supplied level to `0..=8`
  so the reorderer's loops are finite. The rules implemented are the subset of
  UAX #9 named in the documentation — an application that needs the explicit
  embedding rules (X1–X8) or isolate handling should supply levels through
  `set_explicit_levels` and treat this crate's output as the implicit part.

[GitHub security advisories]:
    https://github.com/WyattAu/font-shape/security/advisories/new