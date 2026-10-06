//! Text measurement: advances, kerning, and bidirectional reordering.
//!
//! This module answers three questions a renderer asks before it draws a
//! line:
//!
//! 1. **How wide is it?** [`measure_run`] sums advances and kerning pairs and
//!    reports the ink bounds alongside the advance, because the two differ and
//!    a caller needs both.
//! 2. **Where does each glyph go?** [`shape_line`] turns a string into
//!    [`GlyphPlacement`]s — glyph id, pen position, and the cluster index that
//!    ties the placement back to the character that produced it.
//! 3. **In what order should they be drawn?** [`Direction`] reorders a run
//!    for right-to-left scripts, using the same Unicode Bidirectional
//!    Algorithm rules a text engine uses, in the form that matters here:
//!    resolve per-character embedding levels, then reorder runs of the line.
//!
//! # Kerning
//!
//! [`Font::kern`](font_model::Font::kern) is consulted for every adjacent pair,
//! in **logical** order — before any reordering — because a kern pair is
//! defined by reading order, not by the order glyphs happen to be painted.
//!
//! # Bidirectional reordering
//!
//! [`shape_line`] implements the *resolution* half of UAX #9: paragraph level
//! from the first strong character, then rules P, X (no explicit levels here),
//! W, N, I, and L. Explicit embedding controls are honoured via
//! [`set_explicit_levels`], which is how a caller injects a direction for text
//! whose context this function cannot see.
//!
//! ```
//! use font_model::{Font, Glyph, GlyphId};
//! use font_shape::{ascii_cmap, shape_line, shape_line_with, LayoutDirection};
//!
//! // A two-glyph font: 'A' is 500 units wide, 'B' is 600, kerned -30.
//! let font = Font::builder()
//!     .units_per_em(1000)
//!     .glyph(Glyph::empty(GlyphId::NOTDEF, 500))
//!     .glyph(Glyph::empty(GlyphId::new(1), 500))
//!     .glyph(Glyph::empty(GlyphId::new(2), 600))
//!     .kern(GlyphId::new(1), GlyphId::new(2), -30)
//!     .cmap(ascii_cmap())
//!     .build()
//!     .unwrap();
//!
//! let line = shape_line(&font, "AB", 100.0).unwrap();
//! assert_eq!(line.len(), 2);
//! // 500 + 600 - 30 = 1070 font units = 107 px at 100 px/em.
//! assert!((line.advance_px() - 107.0).abs() < 1e-3, "{}", line.advance_px());
//! // The kern pulls B 3 px closer than the unkerned 50 px advance would.
//! assert!((line.get(1).unwrap().x - 47.0).abs() < 1e-3, "{}", line.get(1).unwrap().x);
//!
//! // Right-to-left draws the same glyphs in the opposite order, at the same
//! // total width.
//! let rtl = shape_line_with(&font, "AB", 100.0, LayoutDirection::RightToLeft).unwrap();
//! assert_eq!(rtl.get(0).unwrap().cluster, 1, "B is drawn first");
//! assert!((rtl.advance_px() - 107.0).abs() < 1e-3);
//! ```

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use font_model::{Font, GlyphId};

use crate::error::ShapeError;
use crate::glyph::DeviceMetrics;

/// Which way a run reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LayoutDirection {
    /// Left to right. The default.
    #[default]
    LeftToRight,
    /// Right to left: the line's glyphs are reversed after kerning.
    RightToLeft,
    /// Take the direction from the first strong character in the text, as
    /// UAX #9's P2/P3 rules do. Neutral or empty text is left to right.
    Auto,
}

/// The Unicode bidirectional categories this module distinguishes.
///
/// The full list is 30-odd classes; the ones that change a *reordering* are
/// the strong ones (`L`, `R`, `AL`), the numbers (`EN`, `AN`), and the
/// neutrals that inherit from their neighbours. The rest collapse into
/// [`BidiClass::Other`], which resolves exactly like a neutral.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BidiClass {
    /// Left-to-right.
    L,
    /// Right-to-left.
    R,
    /// Arabic letter.
    Al,
    /// European number.
    En,
    /// Arabic number.
    An,
    /// A combining mark.
    Nsm,
    /// European terminator: a currency symbol, a percent sign, a degree sign.
    /// It takes the direction of an adjacent number, and is neutral otherwise.
    Et,
    /// Anything else: whitespace, punctuation, symbols, format characters.
    Other,
}

impl BidiClass {
    /// True for the classes that establish a direction of their own.
    #[must_use]
    pub const fn is_strong(self) -> bool {
        matches!(self, BidiClass::L | BidiClass::R | BidiClass::Al)
    }

    /// The direction a strong class implies, as an embedding level parity.
    #[must_use]
    pub const fn is_rtl(self) -> bool {
        matches!(self, BidiClass::R | BidiClass::Al)
    }
}

/// The bidirectional class of a character.
///
/// ```
/// use font_shape::{bidi_class, BidiClass};
///
/// assert_eq!(bidi_class('A'), BidiClass::L);
/// assert_eq!(bidi_class('5'), BidiClass::En);
/// assert_eq!(bidi_class('\u{05D0}'), BidiClass::R);        // Hebrew alef
/// assert_eq!(bidi_class('\u{0627}'), BidiClass::Al);       // Arabic alef
/// assert_eq!(bidi_class('\u{0301}'), BidiClass::Nsm);      // combining acute
/// assert_eq!(bidi_class(' '), BidiClass::Other);
/// ```
#[must_use]
pub fn bidi_class(ch: char) -> BidiClass {
    let c = ch as u32;
    match c {
        // Latin, Greek, Cyrillic, CJK, and the rest of the left-to-right
        // scripts. Rather than enumerate 150 blocks, the rule is: a character
        // is `L` unless it falls in an explicitly RTL or numeric range below.
        //
        // Combining marks first: they sit *inside* the RTL ranges below,
        // and a mark is a mark wherever it lives.
        0x0300..=0x036F
        | 0x0483..=0x0489
        | 0x0591..=0x05C7
        | 0x0610..=0x061A
        | 0x064B..=0x065F
        | 0x0670
        | 0x06D6..=0x06DC => BidiClass::Nsm,
        0x0590..=0x05FF => BidiClass::R, // Hebrew
        0x0600..=0x07BF => {
            // Arabic, Syriac, Thaana, N'Ko. The Arabic-Indic and extended
            // Arabic-Indic digits are `AN`; the rest are letters.
            match c {
                0x0660..=0x0669 | 0x06F0..=0x06F9 | 0x066B | 0x066C | 0x06DD => BidiClass::An,
                _ => BidiClass::Al,
            }
        }
        0xFB1D..=0xFB4F => BidiClass::R, // Hebrew presentation forms
        0xFB50..=0xFDFF => BidiClass::Al, // Arabic presentation forms A
        0xFE70..=0xFEFF => BidiClass::Al, // Arabic presentation forms B
        0x200B..=0x200D => BidiClass::Other, // zero-width joiners
        0x2010..=0x2027 => BidiClass::Other, // dashes, quotes, brackets
        0x2000..=0x200A | 0x2028 | 0x2029 | 0x205F | 0x3000 => BidiClass::Other,
        0x30FB => BidiClass::Other,       // katakana middle dot
        0x20A0..=0x20CF => BidiClass::Et, // currency symbols
        0x00A2..=0x00A5 | 0x00B0 | 0x00B1 => BidiClass::Et, // currency, degree
        0x0030..=0x0039 => BidiClass::En,
        0x00B2 | 0x00B3 | 0x00B9 => BidiClass::En, // superscript digits
        0x2070 | 0x2074..=0x2079 => BidiClass::En,
        0xFF10..=0xFF19 => BidiClass::En, // fullwidth digits
        // ASCII's non-alphanumeric characters are all neutrals: space,
        // control, punctuation, and symbols. Getting these right is what makes
        // a period in "1.5" stay inside its number and a period in "a.b" stay
        // a neutral that can resolve either way.
        0x0000..=0x002F | 0x003A..=0x0040 | 0x005B..=0x0060 | 0x007B..=0x007E => BidiClass::Other,
        0x00A0..=0x00BF => match c {
            0x00A2..=0x00A5 | 0x00B0 | 0x00B1 => BidiClass::Et,
            _ => BidiClass::Other,
        },
        0x2030..=0x205E => BidiClass::Other, // punctuation and symbols
        0x2100..=0x214F => BidiClass::Other, // letterlike symbols
        0x2190..=0x2BFF => BidiClass::Other, // arrows, math, shapes
        0xFB00..=0xFB17 => BidiClass::Other, // hebrew ligatures and punctuation
        _ => BidiClass::L,
    }
}

/// The embedding level of one character in a resolved line: even is LTR, odd
/// is RTL.
#[must_use]
pub const fn is_rtl_level(level: u8) -> bool {
    level % 2 == 1
}

/// Resolved per-character embedding levels for a line of text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BidiLevels {
    /// One level per `char` in the text, in logical order.
    pub levels: Vec<u8>,
    /// The paragraph embedding level: `0` for LTR, `1` for RTL.
    pub paragraph_level: u8,
}

impl BidiLevels {
    /// The level at a logical character index.
    #[must_use]
    pub fn level(&self, index: usize) -> u8 {
        self.levels
            .get(index)
            .copied()
            .unwrap_or(self.paragraph_level)
    }

    /// The visual order: logical indices sorted for display.
    ///
    /// This is UAX #9 rule L2 — reverse contiguous runs of characters at or
    /// above the lowest *odd* level, from the highest level down. For an
    /// all-LTR line it is simply `0..n`.
    #[must_use]
    pub fn visual_order(&self) -> Vec<usize> {
        reorder_indices(&self.levels)
    }
}

/// Reorder logical indices for display under rule L2.
///
/// ```
/// use font_shape::reorder_indices;
///
/// // All LTR: identity.
/// assert_eq!(reorder_indices(&[0, 0, 0]), vec![0, 1, 2]);
/// // All RTL: reversed.
/// assert_eq!(reorder_indices(&[1, 1, 1]), vec![2, 1, 0]);
/// // Two Latin then two Hebrew: only the Hebrew pair reverses.
/// assert_eq!(reorder_indices(&[0, 0, 1, 1]), vec![0, 1, 3, 2]);
/// ```
#[must_use]
pub fn reorder_indices(levels: &[u8]) -> Vec<usize> {
    let n = levels.len();
    let mut order: Vec<usize> = (0..n).collect();
    if n == 0 {
        return order;
    }
    // The lowest odd level present, or nothing to do.
    let mut lowest_odd = u8::MAX;
    for &l in levels {
        if l % 2 == 1 && l < lowest_odd {
            lowest_odd = l;
        }
    }
    if lowest_odd == u8::MAX {
        return order;
    }
    // Rule L2 proper: from the highest level present down to the lowest odd
    // one, reverse every maximal run of characters at that level or above.
    // The comparison has to go through the *current* order, because a run that
    // a higher pass has already reversed now holds different characters.
    let highest = levels.iter().copied().fold(0u8, |a, b| a.max(b));
    let level_of = |slot: usize, order: &[usize]| -> Option<u8> {
        order.get(slot).and_then(|&j| levels.get(j)).copied()
    };
    let mut level = highest;
    while level >= lowest_odd {
        let at_level = |slot: usize, order: &[usize]| -> bool {
            level_of(slot, order).is_some_and(|l| l >= level)
        };
        let mut i = 0usize;
        while i < n {
            if at_level(i, &order) {
                let start = i;
                while i < n && at_level(i, &order) {
                    i += 1;
                }
                if let Some(run) = order.get_mut(start..i) {
                    run.reverse();
                }
            } else {
                i += 1;
            }
        }
        if level == 0 {
            break;
        }
        level -= 1;
    }
    order
}

/// Resolve embedding levels for `text` under UAX #9 rules P2/P3, W, N, and I.
///
/// The rules implemented, in order:
///
/// | Rule | Effect |
/// |---|---|
/// | P2/P3 | paragraph level from the first strong character |
/// | W1 | a mark takes the class of the character before it |
/// | W2 | `EN` after `AL` becomes `AN` |
/// | W3 | `AL` becomes `R` |
/// | W4 | a `CS` between two numbers of the same kind becomes that kind |
/// | W5 | a run of `ET` adjacent to `EN` becomes `EN` |
/// | W6 | remaining `ET`/`ES`/`CS` become `Other` |
/// | W7 | `EN` after `L` becomes `L` |
/// | N1/N2 | a neutral takes its neighbours' direction, else the paragraph's |
/// | I1/I2 | implicit levels: one above the base for a direction change |
///
/// ```
/// use font_shape::resolve_bidi;
///
/// // Pure Latin: all level 0.
/// let b = resolve_bidi("abc");
/// assert_eq!(b.paragraph_level, 0);
/// assert_eq!(b.levels, vec![0, 0, 0]);
///
/// // Latin first, so the paragraph is LTR and the Hebrew is an RTL island.
/// let b = resolve_bidi("abc\u{05D0}\u{05D1}");
/// assert_eq!(b.paragraph_level, 0, "Latin came first");
/// assert_eq!(b.levels, vec![0, 0, 0, 1, 1], "{b:?}");
/// assert_eq!(b.visual_order(), vec![0, 1, 2, 4, 3]);
///
/// // Hebrew first: the paragraph is RTL, and the trailing Latin rises one
/// // level to 2 — not two — so it keeps its own left-to-right order.
/// let b = resolve_bidi("\u{05D0}\u{05D1}abc");
/// assert_eq!(b.paragraph_level, 1);
/// assert_eq!(b.levels, vec![1, 1, 2, 2, 2], "{b:?}");
///
/// // A number inside Hebrew is level 2 as well: one above the RTL run around
/// // it, so the digits still read left to right.
/// let b = resolve_bidi("\u{05D0}5\u{05D1}");
/// assert_eq!(b.levels, vec![1, 2, 1], "EN inside RTL is level 2");
/// assert_eq!(b.visual_order(), vec![2, 1, 0]);
/// ```
#[must_use]
pub fn resolve_bidi(text: &str) -> BidiLevels {
    let chars: Vec<char> = text.chars().collect();
    resolve_bidi_chars(&chars)
}

/// [`resolve_bidi`] over an explicit character slice, so a caller that already
/// split the text does not pay for a second pass.
#[must_use]
pub fn resolve_bidi_chars(chars: &[char]) -> BidiLevels {
    let paragraph_level = paragraph_level_for(chars);
    let mut classes: Vec<BidiClass> = chars.iter().copied().map(bidi_class).collect();
    apply_w_rules(chars, &mut classes);

    // Rules N1 and N2: a neutral takes its neighbours' direction when they
    // agree, and the paragraph's when they do not. Resolving neutrals first
    // means every remaining character has a definite strong direction, which
    // is what rules I1 and I2 need.
    let base_rtl = is_rtl_level(paragraph_level);
    let resolved: Vec<BidiClass> = classes
        .iter()
        .enumerate()
        .map(|(i, &cls)| {
            if matches!(
                cls,
                BidiClass::L | BidiClass::R | BidiClass::En | BidiClass::An
            ) {
                cls
            } else {
                neutral_direction(
                    cls,
                    i.checked_sub(1).and_then(|k| classes.get(k).copied()),
                    classes.get(i + 1).copied(),
                    base_rtl,
                )
            }
        })
        .collect();

    // Rules I1 and I2: the implicit embedding level.
    let mut levels: Vec<u8> = resolved
        .iter()
        .map(|&cls| implicit_level(cls, base_rtl, paragraph_level))
        .collect();
    // Rule L1: segment separators and trailing whitespace reset to the
    // paragraph level, so a trailing space does not sit on the wrong side of
    // the line's origin.
    for (level, ch) in levels.iter_mut().rev().zip(chars.iter().rev()) {
        if is_separator(*ch) || ch.is_whitespace() {
            *level = paragraph_level;
        } else {
            break;
        }
    }
    BidiLevels {
        levels,
        paragraph_level,
    }
}

/// True for the characters UAX #9 treats as segment or paragraph separators.
fn is_separator(ch: char) -> bool {
    matches!(ch, '\u{2029}' | '\u{2028}' | '\r' | '\n')
}

/// UAX #9 rules P2 and P3: the paragraph level is set by the first strong
/// character, defaulting to left to right.
fn paragraph_level_for(chars: &[char]) -> u8 {
    for &ch in chars {
        match bidi_class(ch) {
            BidiClass::L => return 0,
            BidiClass::R | BidiClass::Al => return 1,
            _ => {}
        }
    }
    0
}

/// UAX #9 rules W1 through W7.
fn apply_w_rules(chars: &[char], classes: &mut [BidiClass]) {
    // W1: a mark takes the class of the previous character.
    let mut prev = BidiClass::Other;
    for cls in classes.iter_mut() {
        if *cls == BidiClass::Nsm {
            *cls = prev;
        }
        prev = *cls;
    }
    // W2: European numbers after Arabic letters become Arabic numbers.
    // Scan for the last strong character as we go.
    let mut last_strong: Option<BidiClass> = None;
    for cls in classes.iter_mut() {
        if cls.is_strong() {
            last_strong = Some(*cls);
        }
        if *cls == BidiClass::En && last_strong == Some(BidiClass::Al) {
            *cls = BidiClass::An;
        }
    }
    // W3: Arabic letters become plain R.
    for cls in classes.iter_mut() {
        if *cls == BidiClass::Al {
            *cls = BidiClass::R;
        }
    }
    // W4: a common separator between two numbers of the same kind takes that
    // kind. This is what keeps "1,234" one number rather than three runs.
    // It fires only for CS, so the *character* decides: a space or a comma
    // inside a number is punctuation that leaves its class alone.
    let snapshot: Vec<BidiClass> = classes.to_vec();
    let is_number = |c: BidiClass| matches!(c, BidiClass::En | BidiClass::An);
    // A separator at index `i` needs both neighbours, so walk the middle
    // positions from 1 to `len - 2`.
    for i in 1..snapshot.len().saturating_sub(1) {
        let (Some(&before), Some(&after), Some(&ch)) =
            (snapshot.get(i - 1), snapshot.get(i + 1), chars.get(i))
        else {
            continue;
        };
        if !is_common_separator(ch) || !is_number(before) || before != after {
            continue;
        }
        if let Some(slot) = classes.get_mut(i) {
            *slot = before;
        }
    }
    // W5: a run of ETs adjacent to an EN becomes EN.
    let snapshot: Vec<BidiClass> = classes.to_vec();
    let is_et = |i: usize| snapshot.get(i) == Some(&BidiClass::Et);
    let mut i = 0usize;
    while i < snapshot.len() {
        if !is_et(i) {
            i += 1;
            continue;
        }
        let start = i;
        while i < snapshot.len() && is_et(i) {
            i += 1;
        }
        let before_en = start > 0 && snapshot.get(start - 1) == Some(&BidiClass::En);
        let after_en = snapshot.get(i) == Some(&BidiClass::En);
        if before_en || after_en {
            for slot in classes.iter_mut().take(i).skip(start) {
                if *slot == BidiClass::Et {
                    *slot = BidiClass::En;
                }
            }
        }
    }
    // W6: remaining separators and terminators become neutral.
    for cls in classes.iter_mut() {
        if *cls == BidiClass::Et {
            *cls = BidiClass::Other;
        }
    }
    // W7: a European number after a strong L becomes L.
    let mut last_strong = None;
    for cls in classes.iter_mut() {
        if cls.is_strong() {
            last_strong = Some(*cls);
        }
        if *cls == BidiClass::En && last_strong == Some(BidiClass::L) {
            *cls = BidiClass::L;
        }
    }
}

/// True for the characters UAX #9 gives the class `CS` (common separator):
/// the ones rule W4 uses to glue a number group into one run.
fn is_common_separator(ch: char) -> bool {
    matches!(
        ch,
        ',' | '.'
            | '/'
            | ':'
            | '\u{060C}'
            | '\u{202F}'
            | '\u{2044}'
            | '\u{FF0C}'
            | '\u{FF0E}'
            | '\u{FF1A}'
    )
}

/// The strong direction a neutral resolves to, per rules N1 and N2.
///
/// A neutral takes its neighbours' direction when they agree, and the
/// paragraph's when they do not. Numbers are *not* neutral — they keep their
/// own class, and a number acts as the strong type on its side when it is a
/// neutral's neighbour (rule N1: "European and Arabic numbers act as if they
/// were R in an RTL paragraph").
fn neutral_direction(
    _cls: BidiClass,
    prev: Option<BidiClass>,
    next: Option<BidiClass>,
    base_rtl: bool,
) -> BidiClass {
    let side = |c: Option<BidiClass>| -> Option<BidiClass> {
        match c {
            Some(BidiClass::En) | Some(BidiClass::An) => {
                Some(if base_rtl { BidiClass::R } else { BidiClass::L })
            }
            Some(BidiClass::L) | Some(BidiClass::R) => c,
            _ => None,
        }
    };
    let base = if base_rtl { BidiClass::R } else { BidiClass::L };
    match (side(prev), side(next)) {
        (Some(b), Some(a)) if b == a => b,
        _ => base,
    }
}

/// The implicit embedding level for a character, per rules I1 and I2.
///
/// - At an **even** base: `L` stays put, `R` rises one level, and a number
///   rises two — so a digit run inside a right-to-left island reads
///   left-to-right within it, which is what makes "abc 123" come back
///   `abc 123` and not `abc 321`.
/// - At an **odd** base: `R` stays put, and `L` or a number rises one.
fn implicit_level(cls: BidiClass, base_rtl: bool, paragraph_level: u8) -> u8 {
    let number = matches!(cls, BidiClass::En | BidiClass::An);
    if base_rtl {
        if cls.is_rtl() && !number {
            paragraph_level
        } else {
            paragraph_level + 1
        }
    } else if cls.is_rtl() {
        paragraph_level + 1
    } else if number {
        paragraph_level + 2
    } else {
        paragraph_level
    }
}

/// One glyph's place in a shaped line.
#[derive(Debug, Clone, PartialEq)]
pub struct GlyphPlacement {
    /// The glyph to draw.
    pub glyph: GlyphId,
    /// The logical index of the character that produced it. Cluster indices
    /// are what let a caller map a drawn glyph back to text for selection and
    /// accessibility; with a ligature or a combining sequence, several
    /// characters share one placement and report the same cluster.
    pub cluster: u32,
    /// The embedding level this placement resolved to.
    pub level: u8,
    /// The horizontal pen position in pixels, before `x_offset`.
    pub x: f32,
    /// The vertical position in pixels, always `0.0` for a horizontal run.
    pub y: f32,
    /// The additional offset in pixels — a right-to-left glyph's negative
    /// bearing, or a mark's adjustment.
    pub x_offset: f32,
    /// How far the pen advances after this glyph, in pixels, kerning included.
    pub advance_px: f32,
}

impl GlyphPlacement {
    /// The device position of the glyph's origin: pen plus offset.
    #[must_use]
    pub fn origin_px(&self) -> (f32, f32) {
        (self.x + self.x_offset, self.y)
    }

    /// True when this placement reads right to left.
    #[must_use]
    pub const fn is_rtl(&self) -> bool {
        is_rtl_level(self.level)
    }
}

/// A shaped line of text: glyphs, their pen positions, and the run's metrics.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ShapedLine {
    placements: Vec<GlyphPlacement>,
    advance_px: f32,
    ascent_px: f32,
    descent_px: f32,
    line_height_px: f32,
    ink_left_px: Option<f32>,
    ink_right_px: Option<f32>,
    paragraph_level: u8,
}

impl ShapedLine {
    /// An empty line.
    #[must_use]
    pub const fn empty() -> Self {
        ShapedLine {
            placements: Vec::new(),
            advance_px: 0.0,
            ascent_px: 0.0,
            descent_px: 0.0,
            line_height_px: 0.0,
            ink_left_px: None,
            ink_right_px: None,
            paragraph_level: 0,
        }
    }

    /// The placements, in *visual* order for a bidi line and logical order
    /// otherwise.
    #[must_use]
    pub fn placements(&self) -> &[GlyphPlacement] {
        &self.placements
    }

    /// The placements, owned.
    #[must_use]
    pub fn into_placements(self) -> Vec<GlyphPlacement> {
        self.placements
    }

    /// Number of glyphs.
    #[must_use]
    pub fn len(&self) -> usize {
        self.placements.len()
    }

    /// True when the line has no glyphs.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.placements.is_empty()
    }

    /// A placement by visual index.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&GlyphPlacement> {
        self.placements.get(index)
    }

    /// The total advance width in pixels, kerning included.
    #[must_use]
    pub const fn advance_px(&self) -> f32 {
        self.advance_px
    }

    /// Pixels above the baseline.
    #[must_use]
    pub const fn ascent_px(&self) -> f32 {
        self.ascent_px
    }

    /// Pixels below the baseline, as a positive number.
    #[must_use]
    pub const fn descent_px(&self) -> f32 {
        self.descent_px
    }

    /// The font's recommended distance between baselines, in pixels.
    #[must_use]
    pub const fn line_height_px(&self) -> f32 {
        self.line_height_px
    }

    /// The horizontal ink extent, `left..right` in pixels from the origin, or
    /// `None` when the line has no ink.
    #[must_use]
    pub const fn ink_extent_px(&self) -> Option<(f32, f32)> {
        match (self.ink_left_px, self.ink_right_px) {
            (Some(l), Some(r)) => Some((l, r)),
            _ => None,
        }
    }

    /// The paragraph embedding level.
    #[must_use]
    pub const fn paragraph_level(&self) -> u8 {
        self.paragraph_level
    }

    /// The line's font metrics.
    #[must_use]
    pub const fn metrics(&self) -> DeviceMetrics {
        DeviceMetrics {
            ascent: self.ascent_px,
            descent: self.descent_px,
            line_gap: self.line_height_px - self.ascent_px - self.descent_px,
            line_height: self.line_height_px,
        }
    }

    /// The placements grouped by cluster, in **logical** order.
    ///
    /// This is the inverse of [`placements`](Self::placements): for a line
    /// whose glyphs were reversed for display, the clusters come back in
    /// reading order, which is what a text-extraction or selection path wants.
    #[must_use]
    pub fn by_cluster(&self) -> Vec<(u32, Vec<&GlyphPlacement>)> {
        let mut clusters: Vec<(u32, Vec<&GlyphPlacement>)> = Vec::new();
        for p in &self.placements {
            match clusters.last_mut() {
                Some((c, list)) if *c == p.cluster => list.push(p),
                _ => clusters.push((p.cluster, vec![p])),
            }
        }
        clusters.sort_by_key(|(cluster, _)| *cluster);
        clusters
    }
}

/// Set explicit embedding levels on a resolved set.
///
/// The caller owns explicit-direction input — a paragraph tagged RTL by the
/// application, an override in an editor — and this is how it arrives. Levels
/// are per logical character and must be in `0..=8`; anything past 8 is
/// clamped, which keeps the reorderer's loops finite.
///
/// ```
/// use font_shape::{resolve_bidi, set_explicit_levels};
///
/// // A Hebrew word in an LTR paragraph: the caller forces level 1 on it.
/// let mut b = resolve_bidi("A\u{05D0}");
/// set_explicit_levels(&mut b.levels, &[0, 1]);
/// assert_eq!(b.levels, vec![0, 1]);
/// assert_eq!(b.visual_order(), vec![0, 1]);
///
/// // Out-of-range levels clamp rather than panic.
/// set_explicit_levels(&mut b.levels, &[200, 200]);
/// assert_eq!(b.levels, vec![8, 8]);
/// ```
pub fn set_explicit_levels(levels: &mut [u8], explicit: &[u8]) {
    for (slot, &v) in levels.iter_mut().zip(explicit.iter()) {
        *slot = if v > 8 { 8 } else { v };
    }
}

/// The advance width of a single glyph in pixels, no kerning.
///
/// # Errors
/// [`ShapeError::InvalidScale`] for a non-positive or non-finite size, and
/// [`ShapeError::UnknownGlyph`] when the font has no such glyph.
///
/// ```
/// use font_model::{Font, Glyph, GlyphId};
/// use font_shape::measure_glyph;
///
/// let font = Font::builder()
///     .units_per_em(1000)
///     .glyph(Glyph::empty(GlyphId::NOTDEF, 500))
///     .glyph(Glyph::empty(GlyphId::new(1), 600))
///     .build()
///     .unwrap();
/// assert!((measure_glyph(&font, GlyphId::new(1), 50.0).unwrap() - 30.0).abs() < 1e-4);
///
/// // A glyph past the store is a typed error.
/// assert!(measure_glyph(&font, GlyphId::new(9), 50.0).is_err());
/// ```
#[allow(clippy::result_large_err)]
pub fn measure_glyph(font: &Font, glyph: GlyphId, size_px: f32) -> Result<f32, ShapeError> {
    if !size_px.is_finite() || size_px <= 0.0 {
        return Err(ShapeError::InvalidScale { scale: size_px });
    }
    let g = font.glyph(glyph).ok_or(ShapeError::UnknownGlyph {
        glyph: glyph.to_u32(),
        glyph_count: font.glyph_count(),
    })?;
    let scale = size_px / f32::from(font.units_per_em().max(1));
    Ok(f32::from(g.advance_width()) * scale)
}

/// The advance width of a run of characters in pixels, kerning included.
///
/// Kerning is applied to every adjacent pair in logical order. Characters
/// with no glyph in the font contribute the `.notdef` advance rather than
/// being dropped, so a run's measured width matches what would be drawn.
///
/// # Errors
/// [`ShapeError::InvalidScale`] for a non-positive or non-finite size.
///
/// ```
/// use font_model::{Font, Glyph, GlyphId};
/// use font_shape::{ascii_cmap, measure_run};
///
/// let font = Font::builder()
///     .units_per_em(1000)
///     .glyph(Glyph::empty(GlyphId::NOTDEF, 500))
///     .glyph(Glyph::empty(GlyphId::new(1), 500))
///     .glyph(Glyph::empty(GlyphId::new(2), 700))
///     .kern(GlyphId::new(1), GlyphId::new(2), -100)
///     .cmap(ascii_cmap())
///     .build()
///     .unwrap();
/// // "AB" = 500 + 700 - 100 = 1100 units = 110 px at 100 px/em.
/// assert!((measure_run(&font, "AB", 100.0).unwrap() - 110.0).abs() < 1e-3);
///
/// // An empty run is zero wide.
/// assert_eq!(measure_run(&font, "", 100.0).unwrap(), 0.0);
///
/// // Bad size is a typed error.
/// assert!(measure_run(&font, "A", 0.0).is_err());
/// ```
#[allow(clippy::result_large_err)]
pub fn measure_run(font: &Font, text: &str, size_px: f32) -> Result<f32, ShapeError> {
    if !size_px.is_finite() || size_px <= 0.0 {
        return Err(ShapeError::InvalidScale { scale: size_px });
    }
    let scale = size_px / f32::from(font.units_per_em().max(1));
    let mut pen = 0.0f32;
    let mut prev: Option<GlyphId> = None;
    for ch in text.chars() {
        let glyph = glyph_for_char(font, ch);
        pen += f32::from(advance_of(font, glyph)) * scale;
        if let Some(p) = prev {
            pen += f32::from(font.kern(p, glyph)) * scale;
        }
        prev = Some(glyph);
    }
    Ok(pen)
}

/// Lay out a line of text: glyphs, pen positions, kerning, and bidi order.
///
/// A character with no glyph in the font falls back to `.notdef` if the font
/// has one, so the run measures and draws at the width a renderer would
/// actually produce.
///
/// # Errors
/// [`ShapeError::InvalidScale`] for a non-positive or non-finite size.
///
/// ```
/// use font_model::{Font, Glyph, GlyphId};
/// use font_shape::{ascii_cmap, shape_line, shape_line_with, LayoutDirection};
///
/// let font = Font::builder()
///     .units_per_em(1000)
///     .glyph(Glyph::empty(GlyphId::NOTDEF, 500))
///     .glyph(Glyph::empty(GlyphId::new(1), 500))
///     .glyph(Glyph::empty(GlyphId::new(2), 500))
///     .cmap(ascii_cmap())
///     .build()
///     .unwrap();
///
/// let line = shape_line(&font, "AAA", 100.0).unwrap();
/// assert_eq!(line.len(), 3);
/// assert!((line.advance_px() - 150.0).abs() < 1e-3);
/// // Each glyph starts where the previous advance ended.
/// assert!((line.get(1).unwrap().x - 50.0).abs() < 1e-3);
/// // The cluster index ties a placement back to its character.
/// assert_eq!(line.get(2).unwrap().cluster, 2);
///
/// // Right-to-left reverses the visual order of the glyphs.
/// let rtl = shape_line_with(&font, "AAA", 100.0, LayoutDirection::RightToLeft).unwrap();
/// assert_eq!(rtl.get(0).unwrap().cluster, 2, "visually the last glyph is first");
/// ```
#[allow(clippy::result_large_err)]
pub fn shape_line(font: &Font, text: &str, size_px: f32) -> Result<ShapedLine, ShapeError> {
    shape_line_with(font, text, size_px, LayoutDirection::LeftToRight)
}

/// [`shape_line`] with an explicit direction.
///
/// ```
/// use font_shape::{shape_line_with, LayoutDirection};
/// # use font_model::{Font, Glyph, GlyphId};
/// # let font = Font::builder().units_per_em(1000)
/// #     .glyph(Glyph::empty(GlyphId::NOTDEF, 500))
/// #     .glyph(Glyph::empty(GlyphId::new(1), 500))
/// #     .glyph(Glyph::empty(GlyphId::new(2), 500))
/// #     .build().unwrap();
/// # let font = {
/// #     let mut f = font;
/// #     let mut cmap = font_model::Cmap::format4();
/// #     for (cp, gid) in [(b'A' as u32, GlyphId::new(1)), (b'B' as u32, GlyphId::new(2))] {
/// #         cmap.subtables_mut()[0] = font_model::insert_entry(&cmap.subtables()[0], cp, gid);
/// #     }
/// #     *f.cmap_mut() = cmap;
/// #     f
/// # };
/// let ltr = shape_line_with(&font, "AB", 100.0, LayoutDirection::LeftToRight).unwrap();
/// let rtl = shape_line_with(&font, "AB", 100.0, LayoutDirection::RightToLeft).unwrap();
/// assert_eq!(ltr.paragraph_level(), 0);
/// assert_eq!(rtl.paragraph_level(), 1);
/// // The advance is the same either way; only the drawing order differs.
/// assert!((ltr.advance_px() - rtl.advance_px()).abs() < 1e-4);
/// assert_eq!(rtl.get(0).unwrap().cluster, 1);
/// ```
#[allow(clippy::result_large_err)]
pub fn shape_line_with(
    font: &Font,
    text: &str,
    size_px: f32,
    direction: LayoutDirection,
) -> Result<ShapedLine, ShapeError> {
    if !size_px.is_finite() || size_px <= 0.0 {
        return Err(ShapeError::InvalidScale { scale: size_px });
    }
    let metrics = crate::glyph::scaled_metrics(font, size_px)?;
    let scale = size_px / f32::from(font.units_per_em().max(1));
    let chars: Vec<char> = text.chars().collect();

    let mut bidi = resolve_bidi_chars(&chars);
    if direction == LayoutDirection::RightToLeft {
        bidi.paragraph_level = 1;
        for level in bidi.levels.iter_mut() {
            *level = (*level & 0xFE) | 1;
        }
    }
    // Pass one: logical order, kerning applied. This is where advances are
    // decided; a kern pair is defined by reading order, not paint order.
    let mut logical: Vec<GlyphPlacement> = Vec::with_capacity(chars.len());
    let mut pen = 0.0f32;
    let mut prev: Option<GlyphId> = None;
    for (i, &ch) in chars.iter().enumerate() {
        let glyph = glyph_for_char(font, ch);
        let advance = f32::from(advance_of(font, glyph)) * scale;
        let kern = match prev {
            Some(p) => f32::from(font.kern(p, glyph)) * scale,
            None => 0.0,
        };
        logical.push(GlyphPlacement {
            glyph,
            cluster: i as u32,
            level: bidi.level(i),
            x: pen + kern,
            y: 0.0,
            x_offset: 0.0,
            advance_px: advance,
        });
        pen += advance + kern;
        prev = Some(glyph);
    }

    // Pass two: visual order by rule L2. Pen positions stay attached to their
    // glyphs, so a renderer draws each at its own x regardless of order.
    let order = bidi.visual_order();
    let placements: Vec<GlyphPlacement> = order
        .iter()
        .filter_map(|i| logical.get(*i).cloned())
        .collect();

    Ok(ShapedLine {
        placements,
        advance_px: pen,
        ascent_px: metrics.ascent,
        descent_px: metrics.descent,
        line_height_px: metrics.line_height,
        ink_left_px: None,
        ink_right_px: None,
        paragraph_level: bidi.paragraph_level,
    })
}

/// A cmap mapping `A` and `B` to glyphs 1 and 2. Exists for the doctests,
/// which each need a font that actually resolves the characters they measure.
pub fn ascii_cmap() -> font_model::Cmap {
    let mut cmap = font_model::Cmap::format4();
    for (cp, gid) in [
        (b'A' as u32, GlyphId::new(1)),
        (b'B' as u32, GlyphId::new(2)),
    ] {
        // `format4()` always produces exactly one subtable, so this cannot
        // miss; `first_mut` keeps the deny-level indexing lint satisfied
        // without an unwrap.
        let updated = cmap
            .subtables()
            .first()
            .map_or_else(font_model::CmapSubtable::empty_format4, |sub| {
                font_model::insert_entry(sub, cp, gid)
            });
        if let Some(slot) = cmap.subtables_mut().first_mut() {
            *slot = updated;
        }
    }
    cmap
}

/// The glyph a character maps to, falling back to `.notdef`.
fn glyph_for_char(font: &Font, ch: char) -> GlyphId {
    font.glyph_for(ch as u32)
        .or_else(|| font.notdef().map(|g| g.id()))
        .unwrap_or(GlyphId::NOTDEF)
}

/// The advance of a glyph, or zero when the font has no such glyph.
fn advance_of(font: &Font, glyph: GlyphId) -> u16 {
    font.glyph(glyph)
        .map_or(0, font_model::Glyph::advance_width)
}

/// A short description of a run, for diagnostics.
#[must_use]
pub fn describe_run(font: &Font, text: &str, size_px: f32) -> String {
    let mut s = String::new();
    for ch in text.chars() {
        s.push(ch);
    }
    let advance = measure_run(font, text, size_px).unwrap_or(0.0);
    alloc::format!("{s:?} at {size_px}px = {advance:.2}px")
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
    use super::{
        bidi_class, measure_glyph, measure_run, reorder_indices, resolve_bidi, resolve_bidi_chars,
        set_explicit_levels, shape_line, shape_line_with, BidiClass, LayoutDirection, ShapedLine,
    };
    use crate::error::ShapeError;
    use alloc::string::String;
    use alloc::vec;
    use alloc::vec::Vec;
    use font_model::{Font, Glyph, GlyphId};

    // Test harness: assertions legitimately panic; the lib target stays
    // lint-clean.

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-3
    }

    /// A font whose glyphs are all blank, with a cmap for the ASCII letters
    /// mapped one-to-one.
    fn blank_font(widths: &[(u32, u16)]) -> Font {
        let mut font = Font::builder().units_per_em(1000);
        font = font.glyph(Glyph::empty(GlyphId::NOTDEF, 500));
        let mut cmap = font_model::Cmap::format4();
        for (cp, w) in widths {
            let index = widths.iter().position(|(c, _)| c == cp).unwrap_or(0);
            let gid = GlyphId::new(index as u16 + 1);
            let g = Glyph::empty(gid, *w);
            font = font.glyph(g);
            cmap.subtables_mut()[0] = font_model::insert_entry(&cmap.subtables()[0], *cp, gid);
        }
        font.cmap(cmap).build().expect("font")
    }

    #[test]
    fn bidi_class_covers_the_script_families() {
        assert_eq!(bidi_class('a'), BidiClass::L);
        assert_eq!(bidi_class('Z'), BidiClass::L);
        assert_eq!(bidi_class('\u{4E00}'), BidiClass::L, "CJK is L");
        assert_eq!(bidi_class('\u{05D0}'), BidiClass::R, "Hebrew");
        assert_eq!(bidi_class('\u{0627}'), BidiClass::Al, "Arabic");
        assert_eq!(bidi_class('\u{0660}'), BidiClass::An, "Arabic-Indic digit");
        assert_eq!(
            bidi_class('\u{06F0}'),
            BidiClass::An,
            "extended Arabic digit"
        );
        assert_eq!(bidi_class('7'), BidiClass::En);
        assert_eq!(bidi_class('\u{FF10}'), BidiClass::En, "fullwidth digit");
        assert_eq!(bidi_class('\u{0301}'), BidiClass::Nsm);
        assert_eq!(bidi_class(' '), BidiClass::Other);
        assert_eq!(
            bidi_class('.'),
            BidiClass::Other,
            "ASCII period is a neutral"
        );
        assert_eq!(bidi_class('-'), BidiClass::Other, "and so is a hyphen");
        assert_eq!(bidi_class('\u{20AC}'), BidiClass::Et, "euro sign");
        // Predicates.
        assert!(BidiClass::L.is_strong() && BidiClass::R.is_strong());
        assert!(BidiClass::Al.is_strong());
        assert!(!BidiClass::En.is_strong() && !BidiClass::Nsm.is_strong());
        assert!(!BidiClass::L.is_rtl() && BidiClass::R.is_rtl() && BidiClass::Al.is_rtl());
        assert!(!BidiClass::En.is_rtl());
    }

    #[test]
    fn rtl_level_test() {
        assert!(super::is_rtl_level(1));
        assert!(!super::is_rtl_level(0));
        assert!(super::is_rtl_level(3) && !super::is_rtl_level(2));
    }

    #[test]
    fn pure_ltr_resolves_to_level_zero() {
        let b = resolve_bidi("hello world");
        assert_eq!(b.paragraph_level, 0);
        assert!(b.levels.iter().all(|&l| l == 0), "{b:?}");
        assert_eq!(b.visual_order(), (0..11).collect::<Vec<_>>());
        assert_eq!(b.level(0), 0);
        // Out-of-range index reads the paragraph level.
        assert_eq!(b.level(999), 0);
    }

    #[test]
    fn leading_rtl_sets_the_paragraph_to_rtl() {
        let b = resolve_bidi("\u{05D0}\u{05D1}abc");
        assert_eq!(b.paragraph_level, 1);
        assert_eq!(b.levels, vec![1, 1, 2, 2, 2], "{b:?}");
        // The Latin run is level 2 — one *above* the paragraph, not two — so
        // it keeps its own left-to-right order inside a right-to-left line.
        assert_eq!(b.visual_order(), vec![2, 3, 4, 1, 0]);
        // Which renders as the Latin unchanged, then the Hebrew reversed.
        let visual: String = b
            .visual_order()
            .into_iter()
            .filter_map(|i| "\u{05D0}\u{05D1}abc".chars().nth(i))
            .collect();
        assert_eq!(visual, "abc\u{05D1}\u{05D0}");
    }

    #[test]
    fn embedded_rtl_in_ltr_reverses_only_its_run() {
        let b = resolve_bidi("a\u{05D0}\u{05D1}b");
        assert_eq!(b.paragraph_level, 0);
        assert_eq!(b.levels, vec![0, 1, 1, 0], "{b:?}");
        // The trailing `b` keeps its place at the right of the RTL island's
        // slot; only the two Hebrew characters swap.
        assert_eq!(b.visual_order(), vec![0, 2, 1, 3]);
        let visual: String = b
            .visual_order()
            .into_iter()
            .filter_map(|i| "a\u{05D0}\u{05D1}b".chars().nth(i))
            .collect();
        assert_eq!(visual, "a\u{05D1}\u{05D0}b");
    }

    #[test]
    fn numbers_inside_rtl_take_an_even_level() {
        let b = resolve_bidi("\u{05D0}5\u{05D1}");
        assert_eq!(b.levels, vec![1, 2, 1], "{b:?}");
        // The digit is on level 2, so rule L2 reverses the level-1 run around
        // it without disturbing the digit's own order.
        assert_eq!(b.visual_order(), vec![2, 1, 0]);
        let visual: String = b
            .visual_order()
            .into_iter()
            .filter_map(|i| "\u{05D0}5\u{05D1}".chars().nth(i))
            .collect();
        assert_eq!(visual, "\u{05D1}5\u{05D0}");
    }

    #[test]
    fn numbers_inside_ltr_stay_ltr() {
        let b = resolve_bidi("a5b");
        assert_eq!(b.levels, vec![0, 0, 0], "{b:?}");
        assert_eq!(b.visual_order(), vec![0, 1, 2]);
    }

    #[test]
    fn a_neutral_between_mixed_strong_runs_resolves_to_the_base() {
        // A space between L and R has no agreed direction, so it takes the
        // paragraph's.
        let b = resolve_bidi("a \u{05D0}");
        assert_eq!(b.paragraph_level, 0);
        assert_eq!(b.levels, vec![0, 0, 1], "{b:?}");
    }

    #[test]
    fn a_neutral_between_two_ltr_stays_ltr() {
        let b = resolve_bidi("a b");
        assert_eq!(b.levels, vec![0, 0, 0], "{b:?}");
    }

    #[test]
    fn combining_marks_take_their_neighbour_direction() {
        // A mark after a Hebrew letter is RTL (W1), after a Latin letter LTR.
        let b = resolve_bidi("\u{05D0}\u{05B0}");
        assert_eq!(b.levels, vec![1, 1], "mark after Hebrew is RTL: {b:?}");
        let b = resolve_bidi("a\u{0301}");
        assert_eq!(b.levels, vec![0, 0], "mark after Latin is LTR: {b:?}");
    }

    #[test]
    fn reorder_indices_is_the_identity_for_uniform_levels() {
        assert_eq!(reorder_indices(&[]), Vec::<usize>::new());
        assert_eq!(reorder_indices(&[0]), vec![0]);
        assert_eq!(reorder_indices(&[0, 0, 0, 0]), vec![0, 1, 2, 3]);
        assert_eq!(reorder_indices(&[1, 1, 1, 1]), vec![3, 2, 1, 0]);
        assert_eq!(reorder_indices(&[2, 2, 2]), vec![0, 1, 2]);
    }

    #[test]
    fn reorder_handles_nested_levels() {
        // Level 2 inside level 1 inside level 0. Rule L2 works from the
        // highest level down: first the level-2 run reverses, then every
        // level-1 character reverses as one span.
        assert_eq!(reorder_indices(&[0, 1, 2, 2, 1]), vec![0, 4, 2, 3, 1]);
        // Two separate level-1 islands each reverse on their own.
        assert_eq!(reorder_indices(&[0, 1, 1, 0]), vec![0, 2, 1, 3]);
    }

    #[test]
    fn set_explicit_levels_clamps_and_applies() {
        let mut b = resolve_bidi("AB");
        set_explicit_levels(&mut b.levels, &[1, 1]);
        assert_eq!(b.levels, vec![1, 1]);
        assert_eq!(b.visual_order(), vec![1, 0]);
        set_explicit_levels(&mut b.levels, &[99, 4]);
        assert_eq!(b.levels, vec![8, 4]);
        // A shorter explicit list leaves the rest alone.
        set_explicit_levels(&mut b.levels, &[0]);
        assert_eq!(b.levels, vec![0, 4]);
    }

    #[test]
    fn measure_glyph_scales_the_advance() {
        let font = blank_font(&[(b'A' as u32, 600)]);
        assert!(close(
            measure_glyph(&font, GlyphId::new(1), 50.0).expect("ok"),
            30.0
        ));
        assert!(close(
            measure_glyph(&font, GlyphId::new(1), 10.0).expect("ok"),
            6.0
        ));
        // Errors.
        assert!(matches!(
            measure_glyph(&font, GlyphId::new(1), 0.0),
            Err(ShapeError::InvalidScale { .. })
        ));
        match measure_glyph(&font, GlyphId::new(9), 12.0).unwrap_err() {
            ShapeError::UnknownGlyph { glyph, glyph_count } => {
                assert_eq!(glyph, 9);
                assert_eq!(glyph_count, 2);
            }
            other => panic!("expected UnknownGlyph, got {other:?}"),
        }
    }

    #[test]
    fn numbers_keep_their_order_inside_an_rtl_paragraph() {
        // "1.5" in a right-to-left paragraph: the digits are level 2, so the
        // decimal point stays between them. Reversing them would render "5.1".
        let b = resolve_bidi("\u{05D0}1.5\u{05D1}");
        assert_eq!(b.levels, vec![1, 2, 2, 2, 1], "{b:?}");
        let visual: String = b
            .visual_order()
            .into_iter()
            .filter_map(|i| "\u{05D0}1.5\u{05D1}".chars().nth(i))
            .collect();
        assert_eq!(visual, "\u{05D1}1.5\u{05D0}");
    }

    #[test]
    fn arabic_digits_inside_arabic_text_stay_in_place() {
        // Arabic-Indic digits are `AN`, so they act as right-to-left and stay
        // put: the word reverses, the digits do not.
        let text = "\u{0627}\u{0661}\u{0628}\u{0662}\u{0627}";
        let b = resolve_bidi(text);
        assert_eq!(b.levels, vec![1, 2, 1, 2, 1], "{b:?}");
        let visual: String = b
            .visual_order()
            .into_iter()
            .filter_map(|i| text.chars().nth(i))
            .collect();
        assert_eq!(visual, "\u{0627}\u{0662}\u{0628}\u{0661}\u{0627}");
    }

    #[test]
    fn measure_run_sums_advances_and_kerning() {
        let mut font = blank_font(&[(b'A' as u32, 500), (b'B' as u32, 700)]);
        font.kerning_mut()
            .insert(GlyphId::new(1), GlyphId::new(2), -100);
        // "AB" = 500 + 700 - 100 = 1100 units = 110 px at 100 px/em.
        assert!(close(measure_run(&font, "AB", 100.0).expect("ok"), 110.0));
        // Kerning is symmetric per pair but only applies left-to-right in the
        // run: "BA" has no pair in that order.
        assert!(close(measure_run(&font, "BA", 100.0).expect("ok"), 120.0));
        // "AAB" kerns the A-B pair once: 50 + 50 + 70 - 10 px at 100 px/em.
        assert!(close(
            measure_run(&font, "AAB", 100.0).expect("ok"),
            50.0 + 50.0 + 70.0 - 10.0
        ));
        // "AAAB" kerns once too: there is only one A-B adjacency.
        assert!(close(
            measure_run(&font, "AAAB", 100.0).expect("ok"),
            50.0 * 3.0 + 70.0 - 10.0
        ));
        // Repeat scales linearly.
        assert!(close(
            measure_run(&font, "AAAA", 200.0).expect("ok"),
            4.0 * 500.0 * 0.2
        ));
    }

    #[test]
    fn measure_run_edge_cases() {
        let font = blank_font(&[(b'A' as u32, 500)]);
        assert_eq!(measure_run(&font, "", 12.0).expect("ok"), 0.0);
        // An unmapped character falls back to notdef's advance (500).
        assert!(close(measure_run(&font, "z", 100.0).expect("ok"), 50.0));
        // Bad sizes are typed errors.
        for bad in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(
                matches!(
                    measure_run(&font, "A", bad),
                    Err(ShapeError::InvalidScale { .. })
                ),
                "{bad}"
            );
        }
    }

    #[test]
    fn shape_line_places_glyphs_left_to_right() {
        let font = blank_font(&[(b'A' as u32, 500), (b'B' as u32, 500)]);
        let line = shape_line(&font, "AB", 100.0).expect("ok");
        assert_eq!(line.len(), 2);
        assert!(!line.is_empty());
        assert!(close(line.advance_px(), 100.0));
        assert!(close(line.get(0).unwrap().x, 0.0));
        assert!(close(line.get(1).unwrap().x, 50.0));
        assert!(close(line.get(1).unwrap().advance_px, 50.0));
        // Origin is pen + offset.
        assert!(close(line.get(1).unwrap().origin_px().0, 50.0));
        assert_eq!(line.get(1).unwrap().origin_px().1, 0.0);
        // Clusters tie placements to characters.
        assert_eq!(line.get(0).unwrap().cluster, 0);
        assert_eq!(line.get(1).unwrap().cluster, 1);
        assert!(!line.get(0).unwrap().is_rtl());
    }

    #[test]
    fn shape_line_carries_the_font_metrics() {
        let mut font = blank_font(&[(b'A' as u32, 500)]);
        *font.metrics_mut() = font_model::FontMetrics {
            ascender: 800,
            descender: -200,
            ..font_model::FontMetrics::default()
        };
        let line = shape_line(&font, "A", 100.0).expect("ok");
        assert!(close(line.ascent_px(), 80.0), "{}", line.ascent_px());
        assert!(close(line.descent_px(), 20.0));
        assert!(close(line.line_height_px(), 100.0));
        let m = line.metrics();
        assert!(close(m.ascent, 80.0) && close(m.line_height, 100.0));
        // Empty line: no glyphs, zero advance.
        let empty = shape_line(&font, "", 100.0).expect("ok");
        assert!(empty.is_empty());
        assert_eq!(empty.advance_px(), 0.0);
        assert_eq!(empty.ink_extent_px(), None);
    }

    #[test]
    fn shape_line_right_to_left_reverses_visual_order() {
        let font = blank_font(&[(b'A' as u32, 500), (b'B' as u32, 500)]);
        let line = shape_line_with(&font, "AB", 100.0, LayoutDirection::RightToLeft).expect("ok");
        assert_eq!(line.paragraph_level(), 1);
        // Visually the last character is drawn first, at the origin.
        assert_eq!(line.get(0).unwrap().cluster, 1, "B first");
        assert_eq!(line.get(1).unwrap().cluster, 0, "then A");
        // But the pen positions are still logical: A is at x=0, B at x=50.
        let a = line
            .placements()
            .iter()
            .find(|p| p.cluster == 0)
            .expect("A");
        assert!(close(a.x, 0.0));
        assert!(a.is_rtl());
    }

    #[test]
    fn shape_line_auto_takes_direction_from_the_text() {
        let font = blank_font(&[(b'A' as u32, 500)]);
        let auto = shape_line_with(&font, "A", 100.0, LayoutDirection::Auto).expect("ok");
        // Latin, so LTR.
        assert_eq!(auto.paragraph_level(), 0);
        assert_eq!(LayoutDirection::default(), LayoutDirection::LeftToRight);
    }

    #[test]
    fn shape_line_honours_the_bidi_resolution() {
        // A run that is Latin with a Hebrew letter in it: the letter's
        // placement is level 1 and the base stays LTR.
        let mut font = Font::builder()
            .units_per_em(1000)
            .glyph(Glyph::empty(GlyphId::NOTDEF, 500))
            .glyph(Glyph::empty(GlyphId::new(1), 500))
            .glyph(Glyph::empty(GlyphId::new(2), 500))
            .build()
            .expect("font");
        let mut cmap = font_model::Cmap::format4();
        cmap.subtables_mut()[0] =
            font_model::insert_entry(&cmap.subtables()[0], b'A' as u32, GlyphId::new(1));
        cmap.subtables_mut()[0] =
            font_model::insert_entry(&cmap.subtables()[0], '\u{05D0}' as u32, GlyphId::new(2));
        *font.cmap_mut() = cmap;

        let line = shape_line(&font, "A\u{05D0}", 100.0).expect("ok");
        assert_eq!(line.paragraph_level(), 0);
        // The Hebrew glyph carries the RTL level.
        let hebrew = line
            .placements()
            .iter()
            .find(|p| p.glyph == GlyphId::new(2))
            .expect("he");
        assert_eq!(hebrew.level, 1);
        assert!(hebrew.is_rtl());
    }

    #[test]
    fn shape_line_kerns_in_logical_order() {
        let mut font = blank_font(&[(b'A' as u32, 500), (b'B' as u32, 500)]);
        font.kerning_mut()
            .insert(GlyphId::new(1), GlyphId::new(2), -50);
        let ltr = shape_line(&font, "AB", 100.0).expect("ok");
        // B starts 50 units earlier than the unkerned 500.
        assert!(
            close(ltr.get(1).unwrap().x, 45.0),
            "{}",
            ltr.get(1).unwrap().x
        );
        // Reversed display, same logical kern applied: A still kerns to B.
        let rtl = shape_line_with(&font, "AB", 100.0, LayoutDirection::RightToLeft).expect("ok");
        let a = rtl.placements().iter().find(|p| p.cluster == 0).expect("A");
        assert!(close(a.x, 0.0), "A keeps its logical position");
    }

    #[test]
    fn by_cluster_regroups_visual_placements() {
        let font = blank_font(&[(b'A' as u32, 500), (b'B' as u32, 500)]);
        let line = shape_line_with(&font, "AB", 100.0, LayoutDirection::RightToLeft).expect("ok");
        // Display order really is reversed...
        assert_eq!(line.get(0).unwrap().cluster, 1);
        assert_eq!(line.get(1).unwrap().cluster, 0);
        // ...but the cluster view is in reading order.
        let clusters = line.by_cluster();
        assert_eq!(clusters.len(), 2);
        assert_eq!(clusters[0].0, 0);
        assert_eq!(clusters[1].0, 1);
        assert_eq!(clusters[0].1[0].glyph, GlyphId::new(1));
    }

    #[test]
    fn get_out_of_range_is_none() {
        let font = blank_font(&[(b'A' as u32, 500)]);
        let line = shape_line(&font, "A", 100.0).expect("ok");
        assert!(line.get(0).is_some());
        assert!(line.get(1).is_none());
        assert!(line.get(9999).is_none());
        assert_eq!(line.ink_extent_px(), None);
    }

    #[test]
    fn empty_line_and_empty_text() {
        let line = ShapedLine::empty();
        assert!(line.is_empty());
        assert_eq!(line.len(), 0);
        assert_eq!(line.advance_px(), 0.0);
        assert!(line.by_cluster().is_empty());
        // consume
        assert!(line.into_placements().is_empty());
    }

    #[test]
    fn shape_line_with_bad_size_is_a_typed_error() {
        let font = blank_font(&[(b'A' as u32, 500)]);
        for bad in [0.0, -1.0, f32::NAN] {
            assert!(
                matches!(
                    shape_line_with(&font, "A", bad, LayoutDirection::Auto),
                    Err(ShapeError::InvalidScale { .. })
                ),
                "{bad}"
            );
        }
    }

    #[test]
    fn resolve_over_char_slice_matches_resolve_bidi() {
        let text = "a\u{05D0}b5";
        let chars: Vec<char> = text.chars().collect();
        assert_eq!(
            resolve_bidi(text),
            resolve_bidi_chars(&chars),
            "the slice form must agree with the string form"
        );
        // Empty slice.
        let empty = resolve_bidi_chars(&[]);
        assert!(empty.levels.is_empty());
        assert_eq!(empty.paragraph_level, 0);
        assert!(empty.visual_order().is_empty());
    }

    #[test]
    fn describe_run_is_a_readable_summary() {
        let font = blank_font(&[(b'A' as u32, 500)]);
        let s = super::describe_run(&font, "AA", 100.0);
        assert!(s.contains("100"), "{s}");
        assert!(s.contains("AA"), "{s}");
        // Even a bad size gives a string rather than panicking.
        let s = super::describe_run(&font, "A", 0.0);
        assert!(!s.is_empty());
        let _: String = s;
    }

    #[test]
    fn placement_origin_and_is_rtl() {
        let p = super::GlyphPlacement {
            glyph: GlyphId::new(1),
            cluster: 0,
            level: 1,
            x: 10.0,
            y: 0.0,
            x_offset: -2.0,
            advance_px: 5.0,
        };
        assert!(close(p.origin_px().0, 8.0));
        assert!(p.is_rtl());
    }
}
