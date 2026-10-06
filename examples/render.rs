//! `render` — a preview of what this crate's rasteriser produces.
//!
//! Builds a demo font, rasterizes a line of it at a chosen size, and prints
//! three views of the same glyph:
//!
//! 1. an ASCII coverage map, so the antialiasing is visible in a terminal;
//! 2. a PGM (portable greymap) file, which every image viewer opens;
//! 3. the measured metrics of the line, which is what a layout engine needs.
//!
//! ```text
//! cargo run --example render
//! cargo run --example render -- --size 48 --text "Wg80"
//! cargo run --example render -- --font /path/to/font.ttf
//! cargo run --example render -- --out preview.pgm --png preview.png
//! ```
//!
//! With `--font` the glyphs come from a real typeface through
//! [`font_model::from_sfnt`]; without it, a built-in demo font of geometric
//! shapes is used, so the example runs with no arguments and no assets.
//!
//! `--png` writes a PNG next to the PGM. PNG needs a compressor and a CRC,
//! and this crate deliberately has neither dependency, so the writer is here,
//! in the example, where it can be simple: uncompressed deflate stored blocks
//! wrapped in a CRC-32. The PGM is the real output; the PNG is a convenience.

// The example is a host program: it prints, it writes files, and it unwraps.
#![allow(missing_docs)]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::fs;
use std::io::Write as _;
use std::process::ExitCode;

use font_model::{Font, Glyph, GlyphId, Outline, Path as FontPath};
use font_shape::{
    rasterize_glyph, shape_line_with, GlyphBitmap, Hinting, LayoutDirection, LineCap, LineJoin,
    PathBuilder, StrokeStyle,
};

/// Default em size in pixels.
const DEFAULT_SIZE: f32 = 64.0;

/// The demo text: shapes whose antialiasing shows the rasteriser's behaviour —
/// a diagonal stem, a bowl, a curve, and a hole.
const DEFAULT_TEXT: &str = "ABo8";

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("render: {e}");
            ExitCode::FAILURE
        }
    }
}

/// The parsed command line.
struct Args {
    text: String,
    size: f32,
    font: Option<String>,
    out: Option<String>,
    png: Option<String>,
    hinting: Hinting,
    direction: LayoutDirection,
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let font = match &args.font {
        Some(path) => load_font(path)?,
        None => demo_font(),
    };
    let upem = font.units_per_em();

    println!("font-shape {} — preview", font_shape::VERSION);
    println!(
        "  font     {}",
        args.font.as_deref().unwrap_or("(built-in demo)")
    );
    println!("  em       {upem} units, {upem} -> {:.0} px", args.size);
    println!("  hinting  {:?}", args.hinting);
    println!("  dir      {:?}", args.direction);
    println!("  text     {:?}", args.text);
    println!();

    // ---- Rasterize the run and print it. ----
    let line =
        shape_line_with(&font, &args.text, args.size, args.direction).map_err(|e| e.to_string())?;
    println!(
        "advance {:.1}px  ascent {:.1}px  descent {:.1}px  line-height {:.1}px",
        line.advance_px(),
        line.ascent_px(),
        line.descent_px(),
        line.line_height_px()
    );
    for (cluster, glyphs) in line.by_cluster() {
        let ch = args.text.chars().nth(cluster as usize).unwrap_or(' ');
        for p in &glyphs {
            println!(
                "  cluster {cluster} '{ch}' glyph {} x={:6.1} adv={:5.1} level {}",
                p.glyph.to_u16(),
                p.x,
                p.advance_px,
                p.level
            );
        }
    }
    println!();

    // The frame is the font's ascent plus descender, with a one-pixel margin,
    // and as wide as the run's advance. Anything a glyph draws outside it is
    // clipped, which is what `blit` does.
    let metrics = font_shape::scaled_metrics(&font, args.size).map_err(|e| e.to_string())?;
    // `baseline_row` is where the baseline sits in the frame: one pixel of
    // margin above the ascender.
    let baseline_row = metrics.ascent.ceil() as i32 + 1;
    let height = (baseline_row + metrics.descent.ceil() as i32 + 1).max(1);
    let width = (line.advance_px().ceil() as i32 + 2).max(1);

    // Compose the whole line into one buffer at each glyph's pen position.
    let mut canvas = vec![0u8; (width * height) as usize];
    for p in line.placements() {
        let bmp =
            rasterize_glyph(&font, p.glyph, args.size, args.hinting).map_err(|e| e.to_string())?;
        blit(
            &mut canvas,
            width,
            height,
            &bmp,
            p.x.round() as i32,
            baseline_row,
        );
    }
    print_ascii(&canvas, width, height);
    println!();

    // ---- The stroked demo, which exercises the stroker rather than the
    // glyph path. ----
    println!("stroke demo — a 4-wide round-capped path with round joins:");
    let stroked = demo_stroke();
    let s = font_shape::stroke_path(
        &stroked,
        args.size / 12.0,
        StrokeStyle::new(args.size / 12.0)
            .with_cap(LineCap::Round)
            .with_join(LineJoin::Round),
    );
    println!("  {} commands, {} contours", s.len(), s.subpath_count());
    println!();

    // ---- Write the PGM, and optionally a PNG. ----
    if let Some(path) = &args.out {
        fs::write(path, pgm(&canvas, width, height)).map_err(|e| format!("{path}: {e}"))?;
        println!("wrote {path} ({width}x{height})");
    }
    if let Some(path) = &args.png {
        fs::write(path, png(&canvas, width, height)).map_err(|e| format!("{path}: {e}"))?;
        println!("wrote {path} ({width}x{height})");
    }
    Ok(())
}

/// Parse the command line.
fn parse_args() -> Result<Args, String> {
    let mut text = DEFAULT_TEXT.to_string();
    let mut size = DEFAULT_SIZE;
    let mut font = None;
    let mut out = None;
    let mut png = None;
    let mut hinting = Hinting::None;
    let mut direction = LayoutDirection::Auto;

    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--text" => text = value()?,
            "--size" => {
                let raw = value()?;
                size = raw
                    .parse::<f32>()
                    .map_err(|_| format!("--size {raw} is not a number"))?;
                if !(size.is_finite() && size > 0.0) {
                    return Err(format!("--size {size} is not a positive size"));
                }
            }
            "--font" => font = Some(value()?),
            "--out" => out = Some(value()?),
            "--png" => png = Some(value()?),
            "--grid-fit" => hinting = Hinting::GridFit,
            "--rtl" => direction = LayoutDirection::RightToLeft,
            "-h" | "--help" => {
                println!(
                    "usage: render [--text T] [--size PX] [--font FILE] [--out FILE.pgm] \
                     [--png FILE.png] [--grid-fit] [--rtl]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown flag {other}")),
        }
    }
    Ok(Args {
        text,
        size,
        font,
        out,
        png,
        hinting,
        direction,
    })
}

/// A real font, or the reason it could not be read.
fn load_font(path: &str) -> Result<Font, String> {
    let bytes = fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    font_model::from_sfnt(&bytes).map_err(|e| format!("{path}: {e}"))
}

/// The built-in demo font: four glyphs of pure geometry, mapped to `A`, `B`,
/// `o`, and `8`.
///
/// The shapes are chosen to show the rasteriser rather than to look like
/// letters: a rectangle's edges land exactly on the pixel grid, the triangle
/// has a 45° stem that needs antialiasing on both sides, the ring has a hole
/// that only one fill rule punches, and the diagonal shows the stroker.
fn demo_font() -> Font {
    let mut cmap = font_model::Cmap::format4();
    let mut font = Font::builder()
        .units_per_em(1000)
        .ascender(800)
        .descender(-200)
        .line_gap(100)
        .name(font_model::NameRecord::new(1, "font-shape demo".into()))
        // Glyph ids are their index in the store, so glyph 0 has to come first.
        // `.notdef` is a hollow box here, which is what it is for.
        .glyph(Glyph::new(
            GlyphId::NOTDEF,
            600,
            60,
            ring_outline(60.0, 0.0, 540.0, 700.0, 120.0),
        ));

    for (gid, ch, advance, outline) in [
        (1u16, 'A', 600, box_outline(80.0, 0.0, 520.0, 700.0)),
        (2, 'B', 640, ring_outline(80.0, 0.0, 560.0, 700.0, 200.0)),
        (3, 'o', 560, ring_outline(60.0, 0.0, 500.0, 620.0, 180.0)),
        (4, '8', 560, figure_eight(60.0, 0.0, 500.0, 700.0)),
    ] {
        font = font.glyph(Glyph::new(GlyphId::new(gid), advance, 0, outline));
        // `insert_entry` rebuilds the subtable from its full entry list, so
        // each call adds to the map rather than replacing it.
        let existing = cmap
            .subtables()
            .first()
            .cloned()
            .unwrap_or_else(font_model::CmapSubtable::empty_format4);
        let updated = font_model::insert_entry(&existing, ch as u32, GlyphId::new(gid));
        if let Some(slot) = cmap.subtables_mut().first_mut() {
            *slot = updated;
        } else {
            cmap.subtables_mut().push(updated);
        }
    }
    font = font.cmap(cmap);
    // A little kerning, so the measured line is not just a sum of advances.
    font = font.kern(GlyphId::new(1), GlyphId::new(2), -30);
    font = font.kern(GlyphId::new(3), GlyphId::new(4), -25);
    font.build().expect("the demo font is valid")
}

/// A closed rectangle.
fn box_contour(x0: f32, y0: f32, x1: f32, y1: f32) -> FontPath {
    let mut p = FontPath::starting_at(x0, y0);
    p.line_to(x1, y0);
    p.line_to(x1, y1);
    p.line_to(x0, y1);
    p.close();
    p
}

/// One rectangle as a whole outline.
fn box_outline(x0: f32, y0: f32, x1: f32, y1: f32) -> Outline {
    let mut o = Outline::new();
    o.push_contour(box_contour(x0, y0, x1, y1));
    o
}

/// A rectangular ring: an outer contour and an inner one wound the *opposite*
/// way, which is what a real font does and what the non-zero rule needs to see
/// the hole. Wind them the same way and the ring fills solid — the demo's `o`
/// and `8` are here to show that difference, so they are wound properly.
fn ring_outline(x0: f32, y0: f32, x1: f32, y1: f32, stroke: f32) -> Outline {
    let mut o = box_outline(x0, y0, x1, y1);
    let mut inner = box_contour(x0 + stroke, y0 + stroke, x1 - stroke, y1 - stroke);
    inner.reverse();
    o.push_contour(inner);
    o
}

/// Two stacked bowls, meeting at a waist — the shape of an `8`, and the only
/// demo glyph whose fill needs a fill rule to look right.
fn figure_eight(x0: f32, y0: f32, x1: f32, y1: f32) -> Outline {
    let width = x1 - x0;
    let height = y1 - y0;
    let mut o = ring_outline(x0, y0, x1, y0 + height * 0.56, width * 0.22);
    let lower = ring_outline(x0, y0 + height * 0.44, x1, y1, width * 0.22);
    o.contours_mut().extend(lower.contours().iter().cloned());
    o
}

/// An open path for the stroke demo: a wave with four corners.
fn demo_stroke() -> font_shape::Path2D {
    let mut b = PathBuilder::new();
    b.move_to(0.0, 60.0);
    b.cubic_to(20.0, 0.0, 40.0, 120.0, 60.0, 60.0);
    b.line_to(90.0, 10.0);
    b.line_to(120.0, 60.0);
    b.build()
}

/// Copy a glyph's coverage into the canvas, saturating on overlap.
///
/// `pen_x` is the glyph's pen position and `baseline_row` the canvas row the
/// baseline falls on. A bitmap knows its own placement relative to the
/// baseline — `left` columns to the left of the pen, `top` rows above it — so
/// the two together say where every pixel lands, and a glyph that overhangs is
/// handled rather than assumed away.
fn blit(
    canvas: &mut [u8],
    width: i32,
    height: i32,
    bmp: &GlyphBitmap,
    pen_x: i32,
    baseline_row: i32,
) {
    if bmp.is_empty() {
        return;
    }
    let origin_x = pen_x + bmp.left();
    let origin_y = baseline_row - bmp.top();
    for row in 0..bmp.height() {
        for col in 0..bmp.width() {
            let v = bmp.coverage_at(col, row);
            if v == 0 {
                continue;
            }
            let px = origin_x + col as i32;
            let py = origin_y + row as i32;
            if px < 0 || py < 0 || px >= width || py >= height {
                continue;
            }
            let idx = (py * width + px) as usize;
            if let Some(slot) = canvas.get_mut(idx) {
                *slot = slot.saturating_add(v);
            }
        }
    }
}

/// Print the coverage map as ASCII, one character per pixel.
fn print_ascii(canvas: &[u8], width: i32, height: i32) {
    let ramp = [' ', '.', ':', '-', '=', '+', '*', '#'];
    println!("coverage map {width}x{height}:");
    for y in 0..height {
        let mut line = String::with_capacity(width as usize);
        for x in 0..width {
            let v = canvas.get((y * width + x) as usize).copied().unwrap_or(0);
            // 9 levels of coverage over 0..=255.
            let level = ((u32::from(v) * 8) / 255) as usize;
            line.push(
                *ramp
                    .get(level)
                    .unwrap_or_else(|| ramp.last().expect("non-empty")),
            );
        }
        println!("|{line}|");
    }
}

/// The canvas as a binary PGM (P5).
fn pgm(canvas: &[u8], width: i32, height: i32) -> Vec<u8> {
    let mut out = format!("P5\n{width} {height}\n255\n").into_bytes();
    out.extend_from_slice(canvas);
    out
}

/// The canvas as a greyscale PNG.
///
/// PNG is zlib-deflated, and this crate has no compressor. Deflate's simplest
/// legal encoding is a series of *stored* (uncompressed) blocks, which needs
/// only the two-byte block header and the two-byte adler-32 trailer — plus the
/// zlib wrapper and the PNG chunk CRC-32s. The file is larger than a real PNG
/// and every viewer reads it, which is all an example needs.
fn png(canvas: &[u8], width: i32, height: i32) -> Vec<u8> {
    // Greyscale is colour type 0 with one sample per pixel, so the raw scanline
    // data is the canvas with a filter byte in front of each row.
    let mut raw = Vec::with_capacity(canvas.len() + height as usize);
    for y in 0..height {
        raw.push(0); // filter: none
        let start = (y * width) as usize;
        let end = start + width as usize;
        raw.extend_from_slice(canvas.get(start..end).unwrap_or(&[]));
    }

    let mut out: Vec<u8> = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    // The IHDR must come first: width, height, 8-bit depth, greyscale, then the
    // three "standard" compression/filter/interlace bytes.
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&(width as u32).to_be_bytes());
    ihdr.extend_from_slice(&(height as u32).to_be_bytes());
    ihdr.extend_from_slice(&[8, 0, 0, 0, 0]);
    chunk(&mut out, b"IHDR", &ihdr);
    chunk(&mut out, b"IDAT", &zlib_stored(&raw));
    chunk(&mut out, b"IEND", &[]);
    out
}

/// Append one PNG chunk: length, type, data, CRC.
fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let mut crc_input = Vec::with_capacity(4 + data.len());
    crc_input.extend_from_slice(kind);
    crc_input.extend_from_slice(data);
    out.extend_from_slice(&crc32(&crc_input).to_be_bytes());
}

/// zlib stream around a sequence of stored deflate blocks.
fn zlib_stored(data: &[u8]) -> Vec<u8> {
    let mut out = vec![0x78, 0x01]; // zlib header: deflate, 32K window
                                    // A stored block holds at most 65535 bytes and its length repeats in the
                                    // trailer, so a long canvas needs several.
    let mut chunks: Vec<&[u8]> = Vec::new();
    if data.is_empty() {
        chunks.push(&[]);
    } else {
        chunks.extend(data.chunks(0xffff));
    }
    let count = chunks.len();
    for (i, chunk) in chunks.iter().enumerate() {
        // Only the last stored block sets BFINAL, or a decoder will wait for
        // data that never comes.
        let last = u8::from(i + 1 == count);
        out.push(last); // BFINAL in bit 0, BTYPE 00 (stored) in bits 1-2
        let n = chunk.len() as u16;
        out.extend_from_slice(&n.to_le_bytes());
        out.extend_from_slice(&(!n).to_le_bytes());
        out.extend_from_slice(chunk);
    }
    out.extend_from_slice(&adler32(data).to_be_bytes());
    out
}

/// The CRC-32 PNG chunks carry, over the chunk type and its data.
fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            let mask = 0u32.wrapping_sub(crc & 1);
            crc = (crc >> 1) ^ (0xedb8_8320 & mask);
        }
    }
    !crc
}

/// The Adler-32 the zlib trailer carries.
fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in data {
        a = (a + u32::from(byte)) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

/// Kept so the unused-import lint sees the `Write` trait is genuinely wanted
/// when a caller pipes this somewhere.
#[allow(dead_code)]
fn _write_hint() -> fn(&mut Vec<u8>) -> std::io::Result<()> {
    |v| v.write_all(&[])
}
