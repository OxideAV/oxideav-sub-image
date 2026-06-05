//! PGS / HDMV / Blu-ray `.sup` parser, container, and decoder.
//!
//! A PGS stream is a sequence of display-sets. Each display-set is made
//! of segments, all prefixed by the same on-disk header:
//!
//! ```text
//! "PG" magic (2)
//! PTS (4 BE, 90 kHz units)
//! DTS (4 BE, 90 kHz units)
//! segment type (1): 0x14 PDS, 0x15 ODS, 0x16 PCS, 0x17 WDS, 0x80 END
//! segment size (2 BE)
//! segment body (segment-size bytes)
//! ```
//!
//! A display-set is: one PCS, optional WDS/PDS, one or more ODS,
//! followed by an END segment. The display-set defines a palette + one
//! or more picture objects + the positions at which to compose them on
//! the video canvas.
//!
//! ## Decoder contract
//!
//! The decoder output is one [`oxideav_core::Frame::Video`] per
//! display-set ending with an END segment. The frame's dimensions are
//! the video canvas declared in PCS. The pixel format is
//! [`oxideav_core::PixelFormat::Rgba`]. Transparent background wherever
//! no object is composed; objects are YCbCr→RGB-converted and
//! alpha-composited at the PCS positions.
//!
//! ## Scope / limitations
//!
//! * Decode and encode supported. The encoder crops the input frame to
//!   the tight bounding box of its non-transparent pixels and emits a
//!   single composition object covering that sub-rectangle at the
//!   bbox's `(x, y)`. Colour is quantised into a ≤ 255-entry palette
//!   (index 0 reserved for transparent). When the cropped region has
//!   more than 255 distinct RGBA colours the palette is built from a
//!   3/3/2/2 (R/G/B/A) bucketed reduction, nearest-matching any
//!   surplus colours. Fully-transparent input frames emit an erase
//!   display-set (PCS carrying zero composition objects + empty WDS).
//! * Objects referenced by a composition but not yet seen via ODS are
//!   skipped silently — PGS allows carrying only palette/WDS updates.
//! * ODS fragmentation is handled (an object carries `last_in_sequence`
//!   bits); a PDS version update is treated as a replace.
//! * Cropped compositions (PCS.object_cropped_flag) parse the 8-byte
//!   cropping rectangle that follows the object entry and apply it as a
//!   sub-rectangle selection on the Graphics Object before compositing.
//!   Out-of-range crop coordinates are intersected with the source
//!   object's actual bounds; a crop that lands entirely outside the
//!   object paints nothing, and a zero-extent crop rectangle is rejected
//!   at parse time.

use std::collections::{HashMap, VecDeque};
use std::io::{Read, SeekFrom};

use oxideav_core::{
    CodecId, CodecParameters, CodecResolver, Error, Frame, MediaType, Packet, PixelFormat, Result,
    StreamInfo, TimeBase, VideoFrame, VideoPlane,
};
use oxideav_core::{ContainerRegistry, Demuxer, ProbeData, ProbeScore, ReadSeek};
use oxideav_core::{Decoder, Encoder};

use crate::PGS_CODEC_ID;

// --- segment-type identifiers --------------------------------------------

pub const SEG_PDS: u8 = 0x14;
pub const SEG_ODS: u8 = 0x15;
pub const SEG_PCS: u8 = 0x16;
pub const SEG_WDS: u8 = 0x17;
pub const SEG_END: u8 = 0x80;

// --- composition-state identifiers ---------------------------------------
//
// The PCS carries a 1-byte `composition_state` field that classifies the
// display-set with respect to random access. Three distinct values appear
// in HDMV streams:
//
// * `Normal Case` (0x00) — a per-frame update inside an open epoch. The
//   display-set may rely on palette / object data carried by an earlier
//   set in the same epoch and is *not* a safe seek target on its own.
// * `Acquisition Point` (0x40) — a refresh that carries the full state
//   needed to resume rendering without consulting earlier sets in the
//   epoch. Suitable as a random-access entry point.
// * `Epoch Start` (0x80) — begins a brand-new presentation epoch; all
//   prior state is discarded. Also suitable as a random-access entry
//   point (and the strongest form of one).
//
// Both `Acquisition Point` and `Epoch Start` are random-access points;
// `Normal Case` is not. The [`PgsDemuxer`] uses this distinction to set
// the per-packet `keyframe` flag accurately so seekers can land on a set
// that decodes standalone instead of mid-epoch garbage.
/// PCS composition_state — a per-frame update inside an open epoch.
/// Not a random-access point.
pub const COMP_STATE_NORMAL: u8 = 0x00;
/// PCS composition_state — a refresh carrying the full state needed to
/// resume rendering. A random-access point.
pub const COMP_STATE_ACQUISITION: u8 = 0x40;
/// PCS composition_state — begins a brand-new epoch and discards prior
/// state. The strongest random-access point.
pub const COMP_STATE_EPOCH_START: u8 = 0x80;

/// Returns `true` when the given PCS `composition_state` byte marks a
/// random-access point — i.e. a display-set that decodes standalone
/// without depending on earlier sets in the same epoch. `Acquisition
/// Point` and `Epoch Start` both qualify; `Normal Case` does not.
#[inline]
#[must_use]
pub fn is_random_access(composition_state: u8) -> bool {
    composition_state == COMP_STATE_ACQUISITION || composition_state == COMP_STATE_EPOCH_START
}

/// One parsed display-set segment, carrying only the bytes we need.
#[derive(Clone, Debug)]
pub struct RawSegment {
    pub pts_90k: u32,
    pub dts_90k: u32,
    pub seg_type: u8,
    pub body: Vec<u8>,
}

/// Parse the next raw segment starting at `buf[pos]`. Returns the parsed
/// segment and the new cursor position. `Error::NeedMore` is returned if
/// fewer than 13 bytes remain or the body is truncated — callers reading
/// a complete file typically treat that as EOF.
pub fn read_segment(buf: &[u8], pos: usize) -> Result<(RawSegment, usize)> {
    if pos + 13 > buf.len() {
        return Err(Error::NeedMore);
    }
    if &buf[pos..pos + 2] != b"PG" {
        return Err(Error::invalid("PGS: segment missing 'PG' magic"));
    }
    let pts_90k = u32::from_be_bytes([buf[pos + 2], buf[pos + 3], buf[pos + 4], buf[pos + 5]]);
    let dts_90k = u32::from_be_bytes([buf[pos + 6], buf[pos + 7], buf[pos + 8], buf[pos + 9]]);
    let seg_type = buf[pos + 10];
    let size = u16::from_be_bytes([buf[pos + 11], buf[pos + 12]]) as usize;
    let end = pos + 13 + size;
    if end > buf.len() {
        return Err(Error::NeedMore);
    }
    Ok((
        RawSegment {
            pts_90k,
            dts_90k,
            seg_type,
            body: buf[pos + 13..end].to_vec(),
        },
        end,
    ))
}

// --- PCS ---------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct PresentationComposition {
    pub width: u16,
    pub height: u16,
    pub composition_number: u16,
    /// Random-access classification of this display-set. One of
    /// [`COMP_STATE_NORMAL`], [`COMP_STATE_ACQUISITION`], or
    /// [`COMP_STATE_EPOCH_START`]. Other values pass through verbatim
    /// for callers that want to inspect the raw byte.
    pub composition_state: u8,
    /// Set when the PCS announces that only the palette is being updated;
    /// the object graphics are reused from the previous display-set in
    /// the same epoch. Surfaced here so callers that build their own
    /// render path can short-circuit object reload.
    pub palette_update_flag: bool,
    /// `palette_id` referenced by the composition. Each PDS in the same
    /// display-set is keyed by this id, so callers that need to track
    /// per-id palette state see the current selection here.
    pub palette_id: u8,
    pub objects: Vec<CompositionObject>,
}

#[derive(Clone, Debug)]
pub struct CompositionObject {
    pub object_id: u16,
    pub window_id: u8,
    pub cropped: bool,
    pub forced: bool,
    pub x: u16,
    pub y: u16,
    /// When `cropped` is set, the 8-byte cropping rectangle that follows
    /// the object entry. The rectangle selects a sub-region of the source
    /// graphics object (object-coordinate space) which is then composited
    /// at `(x, y)` on the canvas. Field order matches the rectangle
    /// layout used elsewhere in the BD-ROM HDMV graphics decoder
    /// (`x`, `y`, `w`, `h` as four big-endian `u16`s — the same shape the
    /// WDS window record uses).
    pub crop: Option<CropRect>,
}

/// Sub-rectangle of a Graphics Object selected by a Composition Segment's
/// cropping transform. All four fields are in the source object's
/// coordinate space.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CropRect {
    pub x: u16,
    pub y: u16,
    pub w: u16,
    pub h: u16,
}

/// One window declared by a Window Definition Segment. Coordinates are in
/// canvas-pixel space and a window is the only region a composition object
/// keyed to its `window_id` is permitted to paint into — any object pixels
/// that would land outside this rectangle are dropped at render time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowDefinition {
    pub window_id: u8,
    pub x: u16,
    pub y: u16,
    pub w: u16,
    pub h: u16,
}

/// Parse a Window Definition Segment body into the per-window list. The
/// body layout is `number_of_windows: u8` followed by exactly that many
/// 9-byte records `(window_id: u8, x: u16, y: u16, w: u16, h: u16)`,
/// all big-endian. An empty body, a count that does not match the
/// remaining bytes exactly, or a zero-extent window (`w == 0` or
/// `h == 0`) is rejected — zero-extent is a malformed authoring tool
/// rather than a deliberate "this window paints nothing".
pub fn parse_wds(body: &[u8]) -> Result<Vec<WindowDefinition>> {
    if body.is_empty() {
        return Err(Error::invalid("PGS WDS: empty body"));
    }
    let count = body[0] as usize;
    let expected = 1 + 9 * count;
    if body.len() != expected {
        return Err(Error::invalid(
            "PGS WDS: declared window count does not match body length",
        ));
    }
    let mut windows = Vec::with_capacity(count);
    let mut cur = 1;
    for _ in 0..count {
        let window_id = body[cur];
        let x = u16::from_be_bytes([body[cur + 1], body[cur + 2]]);
        let y = u16::from_be_bytes([body[cur + 3], body[cur + 4]]);
        let w = u16::from_be_bytes([body[cur + 5], body[cur + 6]]);
        let h = u16::from_be_bytes([body[cur + 7], body[cur + 8]]);
        cur += 9;
        if w == 0 || h == 0 {
            return Err(Error::invalid("PGS WDS: window has zero extent"));
        }
        windows.push(WindowDefinition {
            window_id,
            x,
            y,
            w,
            h,
        });
    }
    Ok(windows)
}

fn parse_pcs(body: &[u8]) -> Result<PresentationComposition> {
    if body.len() < 11 {
        return Err(Error::invalid("PGS PCS: body too short"));
    }
    let width = u16::from_be_bytes([body[0], body[1]]);
    let height = u16::from_be_bytes([body[2], body[3]]);
    // body[4] = frame-rate (ignored)
    let composition_number = u16::from_be_bytes([body[5], body[6]]);
    // body[7] = composition_state. Carried through so the demuxer can
    // tell `Normal Case` (mid-epoch update) apart from `Acquisition
    // Point` / `Epoch Start` (random-access entry points). Unknown
    // values pass through unchanged — they're surfaced for downstream
    // consumers rather than rejected.
    let composition_state = body[7];
    // body[8] = palette_update_flag (top bit). The lower bits are
    // reserved; the field is logically a single boolean.
    let palette_update_flag = (body[8] & 0x80) != 0;
    // body[9] = palette_id referenced by this display-set's PDS.
    let palette_id = body[9];
    let n_objects = body[10] as usize;
    let mut cur = 11;
    let mut objects = Vec::with_capacity(n_objects);
    for _ in 0..n_objects {
        if cur + 8 > body.len() {
            return Err(Error::invalid("PGS PCS: object entry truncated"));
        }
        let object_id = u16::from_be_bytes([body[cur], body[cur + 1]]);
        let window_id = body[cur + 2];
        let flags = body[cur + 3];
        let cropped = (flags & 0x80) != 0;
        let forced = (flags & 0x40) != 0;
        let x = u16::from_be_bytes([body[cur + 4], body[cur + 5]]);
        let y = u16::from_be_bytes([body[cur + 6], body[cur + 7]]);
        cur += 8;
        let crop = if cropped {
            if cur + 8 > body.len() {
                return Err(Error::invalid("PGS PCS: cropped object missing crop rect"));
            }
            let cx = u16::from_be_bytes([body[cur], body[cur + 1]]);
            let cy = u16::from_be_bytes([body[cur + 2], body[cur + 3]]);
            let cw = u16::from_be_bytes([body[cur + 4], body[cur + 5]]);
            let ch = u16::from_be_bytes([body[cur + 6], body[cur + 7]]);
            cur += 8;
            // A zero-sized crop is degenerate; flag rather than silently
            // produce an empty paint, because a malformed authoring tool
            // is more likely than a Composition Segment that explicitly
            // wants nothing rendered (an erase display-set is the
            // documented way to clear). The position fields are 16-bit
            // unsigned in object space so checked-arithmetic on the lower
            // bound is automatic; the upper bound is intersected with the
            // referenced object's dimensions at composite time.
            if cw == 0 || ch == 0 {
                return Err(Error::invalid("PGS PCS: crop rectangle has zero extent"));
            }
            Some(CropRect {
                x: cx,
                y: cy,
                w: cw,
                h: ch,
            })
        } else {
            None
        };
        objects.push(CompositionObject {
            object_id,
            window_id,
            cropped,
            forced,
            x,
            y,
            crop,
        });
    }
    Ok(PresentationComposition {
        width,
        height,
        composition_number,
        composition_state,
        palette_update_flag,
        palette_id,
        objects,
    })
}

// --- PDS ---------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Palette {
    /// 256 entries, defaulting to transparent black. Each entry holds
    /// pre-resolved RGBA bytes.
    pub entries: [[u8; 4]; 256],
}

impl Default for Palette {
    fn default() -> Self {
        Self {
            entries: [[0u8; 4]; 256],
        }
    }
}

fn ycbcr_to_rgba(y: u8, cr: u8, cb: u8, a: u8) -> [u8; 4] {
    // BT.601 — PGS uses 8-bit limited-range YCbCr.
    let y = y as i32;
    let cb = cb as i32 - 128;
    let cr = cr as i32 - 128;
    let r = y + ((91881 * cr) >> 16);
    let g = y - ((22554 * cb + 46802 * cr) >> 16);
    let b = y + ((116130 * cb) >> 16);
    [
        r.clamp(0, 255) as u8,
        g.clamp(0, 255) as u8,
        b.clamp(0, 255) as u8,
        a,
    ]
}

fn parse_pds_into(body: &[u8], palette: &mut Palette) -> Result<()> {
    if body.len() < 2 {
        return Err(Error::invalid("PGS PDS: too short"));
    }
    // body[0] = palette_id, body[1] = palette_version. We fold all
    // palettes into one shared table — real streams rarely use more
    // than the id-0 palette per display-set.
    let mut cur = 2;
    while cur + 5 <= body.len() {
        let idx = body[cur] as usize;
        let y = body[cur + 1];
        let cr = body[cur + 2];
        let cb = body[cur + 3];
        let a = body[cur + 4];
        palette.entries[idx] = ycbcr_to_rgba(y, cr, cb, a);
        cur += 5;
    }
    Ok(())
}

// --- ODS ---------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Object {
    pub width: u16,
    pub height: u16,
    /// Indexed-colour pixels (palette ids), row-major.
    pub pixels: Vec<u8>,
}

/// Assemble an ODS (handling the `last_in_sequence` fragmentation bits)
/// into a completed object.
fn parse_ods_into(
    body: &[u8],
    fragments: &mut HashMap<u16, Vec<u8>>,
    objects: &mut HashMap<u16, Object>,
) -> Result<()> {
    if body.len() < 4 {
        return Err(Error::invalid("PGS ODS: header too short"));
    }
    let object_id = u16::from_be_bytes([body[0], body[1]]);
    // body[2] = object_version
    let seq_flag = body[3];
    let first = (seq_flag & 0x80) != 0;
    let last = (seq_flag & 0x40) != 0;

    // First fragment carries: object_data_length (3 BE) + width (2) + height (2) + rle-data.
    // Middle/last fragments carry just rle-data.
    let entry = fragments.entry(object_id).or_default();
    if first {
        entry.clear();
    }
    entry.extend_from_slice(&body[4..]);

    if !last {
        return Ok(());
    }
    // Complete — decode. Pull out width/height + RLE stream.
    let full = fragments.remove(&object_id).unwrap_or_default();
    if full.len() < 7 {
        return Err(Error::invalid(
            "PGS ODS: assembled object data shorter than header",
        ));
    }
    // full[0..3] = object_data_length (u24 BE, informational)
    let width = u16::from_be_bytes([full[3], full[4]]);
    let height = u16::from_be_bytes([full[5], full[6]]);
    if width == 0 || height == 0 {
        return Err(Error::invalid("PGS ODS: zero width/height"));
    }
    let rle = &full[7..];
    let pixels = decode_rle(rle, width as usize, height as usize)?;
    objects.insert(
        object_id,
        Object {
            width,
            height,
            pixels,
        },
    );
    Ok(())
}

/// Decode a PGS run-length encoded bitmap into a width×height indexed
/// buffer. The PGS RLE uses these runs:
///
/// * `CCCCCCCC` (C ≠ 0) — single pixel of colour C.
/// * `00 00`    — end-of-line marker.
/// * `00 LLLLLLLL` (L < 0x40, L > 0) — L pixels of colour 0.
/// * `00 01LLLLLL LLLLLLLL` — (L, 6+8 bits) pixels of colour 0.
/// * `00 10LLLLLL CCCCCCCC` — L pixels of colour C (L < 0x40, L > 0).
/// * `00 11LLLLLL LLLLLLLL CCCCCCCC` — L pixels (14 bits) of colour C.
pub fn decode_rle(rle: &[u8], width: usize, height: usize) -> Result<Vec<u8>> {
    let mut out = vec![0u8; width * height];
    let mut i = 0;
    let mut row = 0usize;
    let mut col = 0usize;
    while i < rle.len() {
        let b0 = rle[i];
        i += 1;
        let (run_len, colour, line_end) = if b0 != 0 {
            (1usize, b0, false)
        } else {
            if i >= rle.len() {
                return Err(Error::invalid("PGS RLE: truncated after 0x00"));
            }
            let b1 = rle[i];
            i += 1;
            if b1 == 0 {
                // end-of-line
                (0usize, 0u8, true)
            } else {
                let hi = b1 & 0xC0;
                let len_lo = (b1 & 0x3F) as usize;
                let (length, colour) = match hi {
                    0x00 => (len_lo, 0u8),
                    0x40 => {
                        if i >= rle.len() {
                            return Err(Error::invalid("PGS RLE: truncated 14-bit length"));
                        }
                        let b2 = rle[i] as usize;
                        i += 1;
                        ((len_lo << 8) | b2, 0u8)
                    }
                    0x80 => {
                        if i >= rle.len() {
                            return Err(Error::invalid("PGS RLE: truncated colour"));
                        }
                        let c = rle[i];
                        i += 1;
                        (len_lo, c)
                    }
                    _ => {
                        if i + 1 >= rle.len() {
                            return Err(Error::invalid("PGS RLE: truncated 14-bit+colour run"));
                        }
                        let b2 = rle[i] as usize;
                        let c = rle[i + 1];
                        i += 2;
                        (((len_lo << 8) | b2), c)
                    }
                };
                (length, colour, false)
            }
        };
        if line_end {
            row += 1;
            col = 0;
            if row > height {
                return Err(Error::invalid("PGS RLE: too many lines"));
            }
            continue;
        }
        if row >= height {
            return Err(Error::invalid("PGS RLE: pixel past end of bitmap"));
        }
        // Clamp runs that would stray past the declared row — malformed
        // streams exist in the wild. Matching a row-exact end is not
        // strictly required: the RLE end-of-line marker is what moves
        // us to the next row.
        let end = col.saturating_add(run_len).min(width);
        let base = row * width + col;
        for px in &mut out[base..base + (end - col)] {
            *px = colour;
        }
        col = end;
    }
    Ok(out)
}

// --- Display-set accumulator + renderer --------------------------------

/// Internal state as we walk segments looking for an END.
#[derive(Default)]
struct DisplaySet {
    pcs: Option<PresentationComposition>,
    // Accepted last-known PTS in 90 kHz units.
    pts_90k: u32,
    palette: Palette,
    object_fragments: HashMap<u16, Vec<u8>>,
    objects: HashMap<u16, Object>,
    /// Window rectangles declared by the WDS, keyed by `window_id`. Each
    /// composition object references one of these; pixels painted outside
    /// the matching rectangle are dropped at render time. Empty when no
    /// WDS has been seen yet — `render` then treats the whole canvas as
    /// paintable, matching the prior behaviour.
    windows: HashMap<u8, WindowDefinition>,
    /// Last-known canvas size (carried over between display-sets when
    /// a PCS-less reset arrives).
    last_canvas: Option<(u16, u16)>,
}

impl DisplaySet {
    fn push(&mut self, seg: &RawSegment) -> Result<()> {
        self.pts_90k = seg.pts_90k;
        match seg.seg_type {
            SEG_PCS => {
                let pcs = parse_pcs(&seg.body)?;
                self.last_canvas = Some((pcs.width, pcs.height));
                self.pcs = Some(pcs);
            }
            SEG_WDS => {
                // Parse the declared windows into a typed table keyed by
                // `window_id` so render-time clipping can look up each
                // composition object's permitted paint area in O(1). A
                // zero-window WDS (the "erase" form) is valid and yields
                // an empty table; `parse_wds` rejects truncated bodies.
                let windows = parse_wds(&seg.body)?;
                self.windows.clear();
                for w in windows {
                    self.windows.insert(w.window_id, w);
                }
            }
            SEG_PDS => {
                parse_pds_into(&seg.body, &mut self.palette)?;
            }
            SEG_ODS => {
                parse_ods_into(&seg.body, &mut self.object_fragments, &mut self.objects)?;
            }
            SEG_END => {}
            _ => {
                // Unknown / reserved segment type — ignored, on the
                // principle that future PGS extensions should not break
                // playback of the parts of the stream we do understand.
            }
        }
        Ok(())
    }

    /// Render the assembled set into an RGBA frame. Returns `Ok(None)`
    /// when the display-set is an "erase" (PCS with zero composition
    /// objects or no PCS at all) — in that case the caller emits a
    /// fully-transparent canvas so the pipeline can clear whatever was
    /// displayed before.
    fn render(&self) -> Result<Option<Vec<u8>>> {
        let Some(pcs) = &self.pcs else {
            return Ok(None);
        };
        let width = pcs.width as usize;
        let height = pcs.height as usize;
        if width == 0 || height == 0 {
            return Err(Error::invalid("PGS PCS: zero-sized canvas"));
        }
        let mut canvas = vec![0u8; width * height * 4];
        for co in &pcs.objects {
            let Some(obj) = self.objects.get(&co.object_id) else {
                continue;
            };
            let ox = co.x as usize;
            let oy = co.y as usize;
            let ow = obj.width as usize;
            let oh = obj.height as usize;
            // When a Composition Segment carries an `object_cropped_flag`,
            // the accompanying cropping rectangle selects a sub-region of
            // the Graphics Object before compositing. Anything outside the
            // crop window is discarded; the surviving sub-rectangle is
            // then placed at the composition `(x, y)` on the canvas.
            // Out-of-range crop coordinates are intersected with the
            // object's actual extent — an authoring tool may declare a
            // crop wider than the object (e.g. for a wipe that runs past
            // the object's right edge); the intersection yields the
            // largest sub-rect that exists on the source bitmap.
            let (sx, sy, sw, sh) = if let Some(c) = co.crop {
                let cx = (c.x as usize).min(ow);
                let cy = (c.y as usize).min(oh);
                let cw = (c.w as usize).min(ow.saturating_sub(cx));
                let ch = (c.h as usize).min(oh.saturating_sub(cy));
                (cx, cy, cw, ch)
            } else {
                (0, 0, ow, oh)
            };
            if sw == 0 || sh == 0 {
                // Crop reduced to nothing after clipping to the object's
                // real bounds — there is no source pixel to paint.
                continue;
            }
            // Intersect the planned paint rectangle on the canvas
            // (`ox..ox+sw`, `oy..oy+sh`) with the window declared for
            // this composition object's `window_id`. PGS windows are
            // the only canvas regions a composition object is allowed
            // to paint into; pixels that would fall outside get dropped
            // by trimming the source slice before the blit. When no
            // WDS has been parsed yet, the table is empty and the
            // whole canvas is treated as paintable — preserving the
            // earlier "no window clipping" behaviour for callers that
            // build display-sets without a WDS.
            let (paint_sx, paint_sy, paint_sw, paint_sh, paint_ox, paint_oy) =
                if let Some(win) = self.windows.get(&co.window_id) {
                    let wx = win.x as usize;
                    let wy = win.y as usize;
                    let ww = win.w as usize;
                    let wh = win.h as usize;
                    // Canvas-space paint rectangle requested by this
                    // composition object.
                    let px0 = ox;
                    let py0 = oy;
                    let px1 = ox.saturating_add(sw);
                    let py1 = oy.saturating_add(sh);
                    // Canvas-space window rectangle.
                    let wx1 = wx.saturating_add(ww);
                    let wy1 = wy.saturating_add(wh);
                    let nx0 = px0.max(wx);
                    let ny0 = py0.max(wy);
                    let nx1 = px1.min(wx1);
                    let ny1 = py1.min(wy1);
                    if nx0 >= nx1 || ny0 >= ny1 {
                        // The composition object lands entirely outside
                        // its declared window — nothing to paint.
                        continue;
                    }
                    let dx = nx0 - px0;
                    let dy = ny0 - py0;
                    let nw = nx1 - nx0;
                    let nh = ny1 - ny0;
                    (sx + dx, sy + dy, nw, nh, nx0, ny0)
                } else {
                    (sx, sy, sw, sh, ox, oy)
                };
            // Composition objects can overlap; when the topmost palette
            // entry is partially transparent the source blends *over*
            // whatever earlier object already painted, rather than
            // overwriting it (Porter–Duff source-over). Build the object
            // (or its cropped sub-rect) as rows of palette indices and
            // run the shared compositor.
            let rows: Vec<Vec<u8>> = (0..paint_sh)
                .map(|row| {
                    let start = (paint_sy + row) * ow + paint_sx;
                    obj.pixels[start..start + paint_sw].to_vec()
                })
                .collect();
            let palette = &self.palette;
            crate::composite::blit_indexed(
                &mut canvas,
                width,
                height,
                &rows,
                paint_ox,
                paint_oy,
                |idx| palette.entries[idx as usize],
            );
        }
        Ok(Some(canvas))
    }
}

// --- Container (.sup) --------------------------------------------------

/// File-extension / container name registration.
pub fn register_container(reg: &mut ContainerRegistry) {
    reg.register_demuxer("pgs", open_pgs);
    reg.register_extension("sup", "pgs");
    reg.register_probe("pgs", probe_pgs);
}

fn probe_pgs(p: &ProbeData) -> ProbeScore {
    // A PGS file opens with the ASCII "PG" magic and reasonable PTS
    // fields. Combined with the .sup extension we give it a high score.
    if p.buf.len() >= 13 && &p.buf[..2] == b"PG" {
        100
    } else if p.ext == Some("sup") {
        25
    } else {
        0
    }
}

fn open_pgs(mut input: Box<dyn ReadSeek>, _codecs: &dyn CodecResolver) -> Result<Box<dyn Demuxer>> {
    let mut buf = Vec::new();
    input.seek(SeekFrom::Start(0))?;
    input.read_to_end(&mut buf)?;

    // Walk the file, gathering segments into display-sets, emitting one
    // packet per display-set with pts = its first segment's PTS.
    let time_base = TimeBase::new(1, 90_000);
    let mut packets: VecDeque<Packet> = VecDeque::new();
    let mut cur = 0usize;
    let mut ds_start_pts: Option<u32> = None;
    let mut ds_buf: Vec<u8> = Vec::new();
    let mut ds_random_access: bool = false;
    let mut last_canvas: Option<(u16, u16)> = None;
    while cur < buf.len() {
        let (seg, next) = match read_segment(&buf, cur) {
            Ok(x) => x,
            Err(_) => break,
        };
        if ds_start_pts.is_none() {
            ds_start_pts = Some(seg.pts_90k);
        }
        ds_buf.extend_from_slice(&buf[cur..next]);
        // Snoop on PCS to capture the canvas size for stream metadata and
        // classify the display-set's random-access state. A set whose PCS
        // marks Acquisition Point or Epoch Start decodes standalone and
        // is a valid seek target; a Normal Case set depends on earlier
        // palette / object data inside the same epoch and is not.
        if seg.seg_type == SEG_PCS {
            if let Ok(pcs) = parse_pcs(&seg.body) {
                last_canvas = Some((pcs.width, pcs.height));
                ds_random_access = is_random_access(pcs.composition_state);
            }
        }
        cur = next;
        if seg.seg_type == SEG_END {
            let pts = ds_start_pts.take().unwrap_or(0);
            let mut packet = Packet::new(0, time_base, std::mem::take(&mut ds_buf));
            packet.pts = Some(pts as i64);
            packet.dts = Some(pts as i64);
            packet.flags.keyframe = ds_random_access;
            packets.push_back(packet);
            // Reset the per-display-set classification before the next
            // set's PCS arrives; a display-set without a PCS (malformed
            // or palette-only) carries no random-access classification.
            ds_random_access = false;
        }
    }
    // Give successive packets a synthetic duration (end - start) and
    // leave the final one's duration unset — we don't know when its cue
    // disappears until another display-set arrives.
    for i in 0..packets.len().saturating_sub(1) {
        let (Some(a), Some(b)) = (packets[i].pts, packets[i + 1].pts) else {
            continue;
        };
        packets[i].duration = Some((b - a).max(0));
    }

    let (w, h) = last_canvas.unwrap_or((0, 0));
    let mut params = CodecParameters::video(CodecId::new(PGS_CODEC_ID));
    params.media_type = MediaType::Subtitle;
    params.width = Some(w as u32);
    params.height = Some(h as u32);
    params.pixel_format = Some(PixelFormat::Rgba);

    let total = packets.back().and_then(|p| p.pts).unwrap_or(0);
    let stream = StreamInfo {
        index: 0,
        time_base,
        duration: Some(total),
        start_time: Some(0),
        params,
    };
    Ok(Box::new(PgsDemuxer {
        streams: [stream],
        packets,
    }))
}

struct PgsDemuxer {
    streams: [StreamInfo; 1],
    packets: VecDeque<Packet>,
}

impl Demuxer for PgsDemuxer {
    fn format_name(&self) -> &str {
        "pgs"
    }

    fn streams(&self) -> &[StreamInfo] {
        &self.streams
    }

    fn next_packet(&mut self) -> Result<Packet> {
        self.packets.pop_front().ok_or(Error::Eof)
    }

    fn duration_micros(&self) -> Option<i64> {
        // PTS is in 90 kHz — µs = pts * (1_000_000 / 90_000) = pts * 100 / 9.
        let pts = self.streams[0].duration?;
        Some(pts * 100 / 9)
    }
}

// --- Decoder -----------------------------------------------------------

/// Build a PGS decoder.
pub fn make_decoder(_params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Ok(Box::new(PgsDecoder {
        codec_id: CodecId::new(PGS_CODEC_ID),
        pending: VecDeque::new(),
        eof: false,
    }))
}

struct PgsDecoder {
    codec_id: CodecId,
    pending: VecDeque<Frame>,
    eof: bool,
}

impl Decoder for PgsDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        let mut ds = DisplaySet::default();
        let mut cur = 0;
        while cur < packet.data.len() {
            let (seg, next) = read_segment(&packet.data, cur)?;
            ds.push(&seg)?;
            cur = next;
        }
        let Some(pcs) = &ds.pcs else {
            // No PCS → nothing to render for this packet. The container
            // normally emits packets only when an END closes a set, so
            // getting here means a malformed or empty display-set.
            return Ok(());
        };
        let width = pcs.width as u32;
        let height = pcs.height as u32;
        let rendered = ds
            .render()?
            .unwrap_or_else(|| vec![0u8; (width as usize) * (height as usize) * 4]);
        let _ = (width, height);
        let frame = VideoFrame {
            pts: packet.pts,
            planes: vec![VideoPlane {
                stride: (pcs.width as usize) * 4,
                data: rendered,
            }],
        };
        self.pending.push_back(Frame::Video(frame));
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        if let Some(f) = self.pending.pop_front() {
            return Ok(f);
        }
        if self.eof {
            Err(Error::Eof)
        } else {
            Err(Error::NeedMore)
        }
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        // PGS is stateless across packets — each display set comes with
        // its own PCS/PDS/ODS chain. Just drop the ready-frame queue and
        // clear the eof latch.
        self.pending.clear();
        self.eof = false;
        Ok(())
    }
}

// --- Encoder -----------------------------------------------------------

/// Build a PGS encoder. The encoder accepts [`Frame::Video`] frames with
/// [`PixelFormat::Rgba`] and emits one [`Packet`] per frame carrying a
/// complete PGS display-set (PCS + WDS + PDS + ODS + END).
///
/// All pixels are quantised into a 255-entry palette (index 0 is reserved
/// for fully-transparent background). When the input has ≤ 255 distinct
/// RGBA colours the quantisation is lossless; otherwise each channel is
/// reduced to a lower bit-depth (3-3-2-2 R-G-B-A) which loses precision
/// but stays within the PGS palette budget.
pub fn make_encoder(_params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    let mut out_params = CodecParameters::video(CodecId::new(PGS_CODEC_ID));
    out_params.media_type = MediaType::Subtitle;
    out_params.pixel_format = Some(PixelFormat::Rgba);
    Ok(Box::new(PgsEncoder {
        codec_id: CodecId::new(PGS_CODEC_ID),
        params: out_params,
        pending: VecDeque::new(),
        composition_number: 0,
    }))
}

struct PgsEncoder {
    codec_id: CodecId,
    params: CodecParameters,
    pending: VecDeque<Packet>,
    composition_number: u16,
}

impl Encoder for PgsEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let Frame::Video(v) = frame else {
            return Err(Error::unsupported(
                "PGS encoder only accepts Frame::Video input",
            ));
        };
        // Pixel format is now stream-level — RGBA is asserted at the
        // CodecParameters layer upstream. Derive (width, height) from
        // the first plane: RGBA stride is `width * 4`.
        if v.planes.is_empty() {
            return Err(Error::invalid("PGS encoder: frame has no plane"));
        }
        let plane = &v.planes[0];
        if plane.stride == 0 || plane.stride % 4 != 0 {
            return Err(Error::invalid(
                "PGS encoder: RGBA plane stride must be a positive multiple of 4",
            ));
        }
        let width = (plane.stride / 4) as u32;
        let height = if width == 0 {
            0
        } else {
            (plane.data.len() / plane.stride) as u32
        };
        if width == 0 || height == 0 {
            return Err(Error::invalid("PGS encoder: zero-sized frame"));
        }
        if self.params.width.is_none() {
            self.params.width = Some(width);
            self.params.height = Some(height);
        }

        let composition_number = self.composition_number;
        self.composition_number = self.composition_number.wrapping_add(1);
        let pts_90k = frame_pts_90k(v).unwrap_or(0);

        // Find the tight bounding box of non-transparent pixels. PGS
        // structurally allows an object smaller than the canvas placed at
        // an arbitrary (x, y); shrinking to the bbox lets a small visible
        // region (a single line of subtitle text near the bottom of a
        // 1920×1080 canvas, for example) be encoded as a small object
        // rather than reserving an RLE run for every transparent row of
        // the full frame.
        let bbox = tight_bbox(v, width as usize, height as usize);
        let bytes = match bbox {
            None => {
                // Fully transparent — emit an erase display-set (PCS with
                // zero objects + empty WDS). The decoder maps this to a
                // fully-transparent canvas, clearing whatever was shown
                // previously.
                encode_erase_display_set(width as u16, height as u16, composition_number, pts_90k)
            }
            Some((bx, by, bw, bh)) => {
                let (indices, palette_rgba) =
                    quantise_rgba_region(v, width as usize, bx, by, bw, bh)?;
                encode_display_set(
                    width as u16,
                    height as u16,
                    bx as u16,
                    by as u16,
                    bw as u16,
                    bh as u16,
                    composition_number,
                    pts_90k,
                    &palette_rgba,
                    &indices,
                )
            }
        };
        let mut packet = Packet::new(0, TimeBase::new(1, 90_000), bytes);
        packet.pts = Some(pts_90k as i64);
        packet.dts = Some(pts_90k as i64);
        packet.flags.keyframe = true;
        self.pending.push_back(packet);
        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        self.pending.pop_front().ok_or(Error::NeedMore)
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}

fn frame_pts_90k(v: &VideoFrame) -> Option<u32> {
    // VideoFrame no longer carries a time_base. Treat the incoming pts as
    // microseconds (the canonical subtitle pts unit) and rescale to the
    // 90 kHz PGS clock.
    let pts = v.pts?;
    let scaled = TimeBase::new(1, 1_000_000).rescale(pts, TimeBase::new(1, 90_000));
    if scaled < 0 {
        Some(0)
    } else {
        Some(scaled as u32)
    }
}

/// Find the smallest rectangle covering every pixel with non-zero alpha.
///
/// Returns `None` if the frame is fully transparent (alpha == 0 everywhere),
/// otherwise `Some((x, y, width, height))` in pixel units. A zero-width
/// or zero-height frame also returns `None`.
fn tight_bbox(v: &VideoFrame, width: usize, height: usize) -> Option<(usize, usize, usize, usize)> {
    if width == 0 || height == 0 || v.planes.is_empty() {
        return None;
    }
    let plane = &v.planes[0];
    let needed = width * 4;
    if plane.stride < needed {
        return None;
    }
    let mut min_x = width;
    let mut max_x: isize = -1;
    let mut min_y = height;
    let mut max_y: isize = -1;
    for row in 0..height {
        let line = &plane.data[row * plane.stride..row * plane.stride + needed];
        let mut row_has = false;
        let mut row_min = width;
        let mut row_max: isize = -1;
        for col in 0..width {
            if line[col * 4 + 3] != 0 {
                if !row_has {
                    row_min = col;
                    row_has = true;
                }
                row_max = col as isize;
            }
        }
        if row_has {
            if (row as isize) < min_y as isize {
                min_y = row;
            }
            if (row as isize) > max_y {
                max_y = row as isize;
            }
            if row_min < min_x {
                min_x = row_min;
            }
            if row_max > max_x {
                max_x = row_max;
            }
        }
    }
    if max_x < 0 || max_y < 0 {
        return None;
    }
    let bw = (max_x as usize - min_x) + 1;
    let bh = (max_y as usize - min_y) + 1;
    Some((min_x, min_y, bw, bh))
}

/// Walk an RGBA frame's sub-rectangle and produce a paired indexed-colour
/// buffer and palette covering just that region. The caller supplies the
/// full frame width (for stride) plus the bbox `(x, y, width, height)`.
/// Index 0 is always fully-transparent black.
fn quantise_rgba_region(
    v: &VideoFrame,
    frame_width: usize,
    bx: usize,
    by: usize,
    bw: usize,
    bh: usize,
) -> Result<(Vec<u8>, Vec<[u8; 4]>)> {
    if v.planes.is_empty() {
        return Err(Error::invalid("PGS encoder: RGBA frame has no plane"));
    }
    let plane = &v.planes[0];
    let needed = frame_width * 4;
    if plane.stride < needed {
        return Err(Error::invalid("PGS encoder: RGBA stride too small"));
    }
    if bx + bw > frame_width || by * plane.stride + needed > plane.data.len() {
        return Err(Error::invalid("PGS encoder: bbox out of frame"));
    }
    let mut palette: Vec<[u8; 4]> = Vec::with_capacity(256);
    palette.push([0, 0, 0, 0]);
    let mut map: HashMap<[u8; 4], u8> = HashMap::new();
    map.insert([0, 0, 0, 0], 0);

    let mut indices = vec![0u8; bw * bh];
    let mut quantise_harder = false;
    'scan: for row in 0..bh {
        let src_row = by + row;
        let line = &plane.data[src_row * plane.stride..src_row * plane.stride + needed];
        for col in 0..bw {
            let src_col = bx + col;
            let px = &line[src_col * 4..src_col * 4 + 4];
            let key = if px[3] == 0 {
                [0, 0, 0, 0]
            } else {
                [px[0], px[1], px[2], px[3]]
            };
            if let Some(&idx) = map.get(&key) {
                indices[row * bw + col] = idx;
                continue;
            }
            if palette.len() >= 255 {
                quantise_harder = true;
                break 'scan;
            }
            let idx = palette.len() as u8;
            palette.push(key);
            map.insert(key, idx);
            indices[row * bw + col] = idx;
        }
    }

    if quantise_harder {
        return quantise_rgba_332_region(v, frame_width, bx, by, bw, bh);
    }
    Ok((indices, palette))
}

/// Fallback quantisation: bucket R/G/B to 3/3/2 bits and A to 2 bits over
/// a sub-rectangle of the frame. Yields up to 256 distinct indices;
/// index 0 is fully-transparent.
fn quantise_rgba_332_region(
    v: &VideoFrame,
    frame_width: usize,
    bx: usize,
    by: usize,
    bw: usize,
    bh: usize,
) -> Result<(Vec<u8>, Vec<[u8; 4]>)> {
    let plane = &v.planes[0];
    let needed = frame_width * 4;
    if plane.stride < needed {
        return Err(Error::invalid("PGS encoder: RGBA stride too small"));
    }
    if bx + bw > frame_width {
        return Err(Error::invalid("PGS encoder: bbox out of frame"));
    }
    let mut palette: Vec<[u8; 4]> = Vec::with_capacity(256);
    palette.push([0, 0, 0, 0]);
    let mut map: HashMap<[u8; 4], u8> = HashMap::new();
    map.insert([0, 0, 0, 0], 0);
    let mut indices = vec![0u8; bw * bh];
    for row in 0..bh {
        let src_row = by + row;
        let line = &plane.data[src_row * plane.stride..src_row * plane.stride + needed];
        for col in 0..bw {
            let src_col = bx + col;
            let px = &line[src_col * 4..src_col * 4 + 4];
            if px[3] == 0 {
                indices[row * bw + col] = 0;
                continue;
            }
            let r = px[0] & 0xE0;
            let g = px[1] & 0xE0;
            let b = px[2] & 0xC0;
            let a = match px[3] {
                0..=63 => 0x3F,
                64..=127 => 0x7F,
                128..=191 => 0xBF,
                _ => 0xFF,
            };
            let key = [r, g, b, a];
            if let Some(&idx) = map.get(&key) {
                indices[row * bw + col] = idx;
                continue;
            }
            let idx = palette.len() as u8;
            palette.push(key);
            map.insert(key, idx);
            indices[row * bw + col] = idx;
            if palette.len() == 256 {
                // No room for a further colour — subsequent novel pixels
                // snap to the nearest existing entry.
                for row2 in row..bh {
                    let src_row2 = by + row2;
                    let line2 =
                        &plane.data[src_row2 * plane.stride..src_row2 * plane.stride + needed];
                    let start_col = if row2 == row { col + 1 } else { 0 };
                    for col2 in start_col..bw {
                        let src_col2 = bx + col2;
                        let px2 = &line2[src_col2 * 4..src_col2 * 4 + 4];
                        let key2 = if px2[3] == 0 {
                            [0, 0, 0, 0]
                        } else {
                            let r = px2[0] & 0xE0;
                            let g = px2[1] & 0xE0;
                            let b = px2[2] & 0xC0;
                            let a = match px2[3] {
                                0..=63 => 0x3F,
                                64..=127 => 0x7F,
                                128..=191 => 0xBF,
                                _ => 0xFF,
                            };
                            [r, g, b, a]
                        };
                        indices[row2 * bw + col2] = *map
                            .get(&key2)
                            .unwrap_or(&nearest_palette_entry(&palette, key2));
                    }
                }
                return Ok((indices, palette));
            }
        }
    }
    Ok((indices, palette))
}

fn nearest_palette_entry(palette: &[[u8; 4]], key: [u8; 4]) -> u8 {
    let mut best = 0u8;
    let mut best_d = i32::MAX;
    for (i, entry) in palette.iter().enumerate() {
        let dr = entry[0] as i32 - key[0] as i32;
        let dg = entry[1] as i32 - key[1] as i32;
        let db = entry[2] as i32 - key[2] as i32;
        let da = entry[3] as i32 - key[3] as i32;
        let d = dr * dr + dg * dg + db * db + da * da;
        if d < best_d {
            best_d = d;
            best = i as u8;
        }
    }
    best
}

/// Encode an indexed bitmap + palette into a single PGS display-set
/// byte-string (PCS + WDS + PDS + ODS + END, each with the shared
/// "PG" segment header). The composition object sits at
/// `(obj_x, obj_y)` on a `canvas_w × canvas_h` canvas and itself
/// measures `obj_w × obj_h` pixels — `indices` must be `obj_w * obj_h`
/// bytes long.
#[allow(clippy::too_many_arguments)]
fn encode_display_set(
    canvas_w: u16,
    canvas_h: u16,
    obj_x: u16,
    obj_y: u16,
    obj_w: u16,
    obj_h: u16,
    composition_number: u16,
    pts_90k: u32,
    palette: &[[u8; 4]],
    indices: &[u8],
) -> Vec<u8> {
    let mut out = Vec::new();

    // PCS.
    let mut pcs = Vec::new();
    pcs.extend_from_slice(&canvas_w.to_be_bytes());
    pcs.extend_from_slice(&canvas_h.to_be_bytes());
    pcs.push(0x10); // frame rate (ignored by decoders but conventional)
    pcs.extend_from_slice(&composition_number.to_be_bytes());
    pcs.push(0x80); // composition state: epoch-start
    pcs.push(0); // palette update flag
    pcs.push(0); // palette id
    pcs.push(1); // one composition object
    pcs.extend_from_slice(&1u16.to_be_bytes()); // object id
    pcs.push(0); // window id
    pcs.push(0); // flags (not cropped, not forced)
    pcs.extend_from_slice(&obj_x.to_be_bytes()); // x
    pcs.extend_from_slice(&obj_y.to_be_bytes()); // y
    push_segment(&mut out, pts_90k, SEG_PCS, &pcs);

    // WDS — declare a single window matching the object's footprint.
    // Decoders use this to scope the area that will be redrawn between
    // display-sets; clamp to canvas if the object would overrun (a
    // 1-pixel-wide object at canvas_w-1 still fits).
    let win_w = obj_w.min(canvas_w.saturating_sub(obj_x));
    let win_h = obj_h.min(canvas_h.saturating_sub(obj_y));
    let mut wds = Vec::new();
    wds.push(1); // one window
    wds.push(0); // window id
    wds.extend_from_slice(&obj_x.to_be_bytes()); // x
    wds.extend_from_slice(&obj_y.to_be_bytes()); // y
    wds.extend_from_slice(&win_w.to_be_bytes());
    wds.extend_from_slice(&win_h.to_be_bytes());
    push_segment(&mut out, pts_90k, SEG_WDS, &wds);

    // PDS.
    let mut pds = Vec::new();
    pds.push(0); // palette id
    pds.push(0); // palette version
    for (idx, rgba) in palette.iter().enumerate() {
        if idx >= 256 {
            break;
        }
        if rgba[3] == 0 {
            // PGS does carry transparent entries, but the canonical form
            // for "unused" is alpha 0 with zeroed chroma — emit it.
            pds.push(idx as u8);
            pds.push(0);
            pds.push(0x80);
            pds.push(0x80);
            pds.push(0);
            continue;
        }
        let (y, cb, cr) = rgb_to_ycbcr_bt601(rgba[0], rgba[1], rgba[2]);
        pds.push(idx as u8);
        pds.push(y);
        pds.push(cr);
        pds.push(cb);
        pds.push(rgba[3]);
    }
    push_segment(&mut out, pts_90k, SEG_PDS, &pds);

    // ODS — single fragment (0xC0 = first + last sequence).
    let rle = encode_rle(indices, obj_w as usize, obj_h as usize);
    let mut ods = Vec::new();
    ods.extend_from_slice(&1u16.to_be_bytes()); // object id
    ods.push(0); // object version
    ods.push(0xC0); // first + last
    let obj_data_len = (rle.len() + 4) as u32; // width+height (4) + rle
    ods.push(((obj_data_len >> 16) & 0xFF) as u8);
    ods.push(((obj_data_len >> 8) & 0xFF) as u8);
    ods.push((obj_data_len & 0xFF) as u8);
    ods.extend_from_slice(&obj_w.to_be_bytes());
    ods.extend_from_slice(&obj_h.to_be_bytes());
    ods.extend_from_slice(&rle);
    push_segment(&mut out, pts_90k, SEG_ODS, &ods);

    // END.
    push_segment(&mut out, pts_90k, SEG_END, &[]);

    out
}

/// Encode an "erase" display-set — a PCS carrying zero composition
/// objects, an empty WDS, and the END marker. This is the canonical
/// representation of a fully-transparent frame (no objects to compose),
/// and tells the decoder/player to clear whatever subtitle was on
/// screen before.
fn encode_erase_display_set(
    canvas_w: u16,
    canvas_h: u16,
    composition_number: u16,
    pts_90k: u32,
) -> Vec<u8> {
    let mut out = Vec::new();

    let mut pcs = Vec::new();
    pcs.extend_from_slice(&canvas_w.to_be_bytes());
    pcs.extend_from_slice(&canvas_h.to_be_bytes());
    pcs.push(0x10); // frame rate
    pcs.extend_from_slice(&composition_number.to_be_bytes());
    pcs.push(0x80); // composition state: epoch-start (forces redraw)
    pcs.push(0); // palette update flag
    pcs.push(0); // palette id
    pcs.push(0); // zero composition objects
    push_segment(&mut out, pts_90k, SEG_PCS, &pcs);

    // WDS — zero windows. The body byte count is one (the window-count
    // field). An entirely empty body is rejected by `DisplaySet::push`,
    // so always emit the count.
    let wds = vec![0u8]; // zero windows
    push_segment(&mut out, pts_90k, SEG_WDS, &wds);

    // END.
    push_segment(&mut out, pts_90k, SEG_END, &[]);

    out
}

fn push_segment(out: &mut Vec<u8>, pts_90k: u32, seg_type: u8, body: &[u8]) {
    out.extend_from_slice(b"PG");
    out.extend_from_slice(&pts_90k.to_be_bytes());
    out.extend_from_slice(&0u32.to_be_bytes()); // DTS — not used
    out.push(seg_type);
    out.extend_from_slice(&(body.len() as u16).to_be_bytes());
    out.extend_from_slice(body);
}

fn rgb_to_ycbcr_bt601(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let r = r as i32;
    let g = g as i32;
    let b = b as i32;
    let y = ((77 * r + 150 * g + 29 * b + 128) >> 8) as u8;
    let cb = (((-43 * r - 84 * g + 127 * b + 128) >> 8) + 128).clamp(0, 255) as u8;
    let cr = (((127 * r - 106 * g - 21 * b + 128) >> 8) + 128).clamp(0, 255) as u8;
    (y, cb, cr)
}

/// Encode a width×height indexed-colour buffer into PGS run-length form
/// (see [`decode_rle`] for the inverse description).
pub fn encode_rle(pixels: &[u8], width: usize, height: usize) -> Vec<u8> {
    debug_assert_eq!(pixels.len(), width * height);
    // Pre-size the output. A worst-case run-of-singletons emits one byte
    // per pixel plus a 2-byte end-of-line per row; a best-case long-run
    // input emits 3-4 bytes per row. Allocating once at the
    // singleton-worst-case bound avoids growth churn for typical
    // subtitle-text bitmaps without overcommitting on the long-run path,
    // which lands inside the initial capacity anyway.
    let mut out = Vec::with_capacity(pixels.len() + height * 2);
    for row in 0..height {
        let mut col = 0usize;
        while col < width {
            let colour = pixels[row * width + col];
            let mut run = 1usize;
            while col + run < width && pixels[row * width + col + run] == colour && run < 0x3FFF {
                run += 1;
            }
            emit_run(&mut out, run, colour);
            col += run;
        }
        // End-of-line marker: 00 00.
        out.push(0);
        out.push(0);
    }
    out
}

fn emit_run(out: &mut Vec<u8>, run: usize, colour: u8) {
    if colour == 0 {
        // Transparent background run.
        if run < 64 {
            out.push(0);
            out.push(run as u8); // 00LLLLLL — L > 0
        } else {
            out.push(0);
            out.push(0x40 | ((run >> 8) & 0x3F) as u8);
            out.push((run & 0xFF) as u8);
        }
        return;
    }
    if run == 1 {
        out.push(colour);
        return;
    }
    if run < 4 {
        // Short runs are cheaper as N back-to-back single pixels.
        for _ in 0..run {
            out.push(colour);
        }
        return;
    }
    if run < 64 {
        out.push(0);
        out.push(0x80 | (run as u8 & 0x3F));
        out.push(colour);
    } else {
        out.push(0);
        out.push(0xC0 | ((run >> 8) & 0x3F) as u8);
        out.push((run & 0xFF) as u8);
        out.push(colour);
    }
}

// --- Test helper -------------------------------------------------------

/// Build a minimal PGS display-set carrying one PCS, one PDS, one ODS,
/// and the terminating END. Used by this crate's tests. Public so that
/// external integration tests can exercise the decoder without pulling
/// in `oxideav-tests`.
#[doc(hidden)]
pub fn build_demo_display_set(
    canvas: (u16, u16),
    object: (u16, u16),
    position: (u16, u16),
    palette: &[(u8, [u8; 4])],
    pixels: &[u8],
) -> Vec<u8> {
    fn segment(out: &mut Vec<u8>, pts_90k: u32, seg_type: u8, body: &[u8]) {
        out.extend_from_slice(b"PG");
        out.extend_from_slice(&pts_90k.to_be_bytes());
        out.extend_from_slice(&0u32.to_be_bytes());
        out.push(seg_type);
        out.extend_from_slice(&(body.len() as u16).to_be_bytes());
        out.extend_from_slice(body);
    }

    let mut out = Vec::new();

    // PCS.
    let mut pcs = Vec::new();
    pcs.extend_from_slice(&canvas.0.to_be_bytes());
    pcs.extend_from_slice(&canvas.1.to_be_bytes());
    pcs.push(0); // frame rate
    pcs.extend_from_slice(&1u16.to_be_bytes()); // composition number
    pcs.push(0); // composition state (normal)
    pcs.push(0); // palette update flag
    pcs.push(0); // palette id
    pcs.push(1); // one composition object
    pcs.extend_from_slice(&1u16.to_be_bytes()); // object id
    pcs.push(0); // window id
    pcs.push(0); // flags
    pcs.extend_from_slice(&position.0.to_be_bytes());
    pcs.extend_from_slice(&position.1.to_be_bytes());
    segment(&mut out, 0, SEG_PCS, &pcs);

    // WDS (one window covering the full canvas).
    let mut wds = Vec::new();
    wds.push(1); // one window
    wds.push(0); // window id
    wds.extend_from_slice(&0u16.to_be_bytes()); // x
    wds.extend_from_slice(&0u16.to_be_bytes()); // y
    wds.extend_from_slice(&canvas.0.to_be_bytes());
    wds.extend_from_slice(&canvas.1.to_be_bytes());
    segment(&mut out, 0, SEG_WDS, &wds);

    // PDS.
    let mut pds = Vec::new();
    pds.push(0); // palette id
    pds.push(0); // palette version
    for (idx, rgba) in palette {
        // Convert to YCbCr.
        let r = rgba[0] as i32;
        let g = rgba[1] as i32;
        let b = rgba[2] as i32;
        let y = ((77 * r + 150 * g + 29 * b + 128) >> 8) as u8;
        let cb = (((-43 * r - 84 * g + 127 * b + 128) >> 8) + 128).clamp(0, 255) as u8;
        let cr = (((127 * r - 106 * g - 21 * b + 128) >> 8) + 128).clamp(0, 255) as u8;
        pds.push(*idx);
        pds.push(y);
        pds.push(cr);
        pds.push(cb);
        pds.push(rgba[3]);
    }
    segment(&mut out, 0, SEG_PDS, &pds);

    // ODS (first+last sequence). Single-pixel runs + end-of-line, shared
    // with the fragmented builder via `demo_rle`.
    let rle = demo_rle(object, pixels);
    let mut ods = Vec::new();
    ods.extend_from_slice(&1u16.to_be_bytes()); // object id
    ods.push(0); // object version
    ods.push(0xC0); // first + last sequence
    let obj_data_len = (rle.len() + 4) as u32; // width+height (4) + rle
    ods.push(((obj_data_len >> 16) & 0xFF) as u8);
    ods.push(((obj_data_len >> 8) & 0xFF) as u8);
    ods.push((obj_data_len & 0xFF) as u8);
    ods.extend_from_slice(&object.0.to_be_bytes());
    ods.extend_from_slice(&object.1.to_be_bytes());
    ods.extend_from_slice(&rle);
    segment(&mut out, 0, SEG_ODS, &ods);

    // END.
    segment(&mut out, 0, SEG_END, &[]);
    out
}

/// Encode `pixels` (one byte per palette index, row-major) into the
/// single-pixel-run + end-of-line RLE form `build_demo_display_set`
/// uses. Shared by the single- and multi-fragment demo builders so both
/// carry byte-identical object data.
fn demo_rle(object: (u16, u16), pixels: &[u8]) -> Vec<u8> {
    let w = object.0 as usize;
    let h = object.1 as usize;
    let mut rle = Vec::new();
    for row in 0..h {
        for c in 0..w {
            let p = pixels[row * w + c];
            if p == 0 {
                // short run-of-zero form: 00 01 (1 pixel of colour 0).
                rle.push(0);
                rle.push(1);
            } else {
                rle.push(p);
            }
        }
        rle.push(0);
        rle.push(0);
    }
    rle
}

/// Like [`build_demo_display_set`], but splits the single object's data
/// across `fragments` ODS segments using the PGS sequence flags
/// (first / continuation / last). A real PGS muxer fragments any object
/// whose data exceeds the per-segment size limit; `build_demo_display_set`
/// only ever emits a first+last (single-segment) object, so this builder
/// is what exercises the decoder's reassembly path.
///
/// The first fragment carries the object_data_length + width + height
/// header followed by the leading slice of RLE; continuation fragments
/// carry raw RLE only; the final fragment sets the `last_in_sequence`
/// flag. `fragments` is clamped to at least 1. The split points fall at
/// even byte boundaries of the *whole* `(header ++ rle)` payload, so a
/// boundary can land inside the 7-byte header, confirming the decoder
/// concatenates fragments before interpreting any field.
#[doc(hidden)]
pub fn build_demo_display_set_fragmented(
    canvas: (u16, u16),
    object: (u16, u16),
    position: (u16, u16),
    palette: &[(u8, [u8; 4])],
    pixels: &[u8],
    fragments: usize,
) -> Vec<u8> {
    fn segment(out: &mut Vec<u8>, pts_90k: u32, seg_type: u8, body: &[u8]) {
        out.extend_from_slice(b"PG");
        out.extend_from_slice(&pts_90k.to_be_bytes());
        out.extend_from_slice(&0u32.to_be_bytes());
        out.push(seg_type);
        out.extend_from_slice(&(body.len() as u16).to_be_bytes());
        out.extend_from_slice(body);
    }

    let fragments = fragments.max(1);

    // Reuse build_demo_display_set to emit the PCS / WDS / PDS prefix
    // verbatim, then splice our fragmented ODS chain in place of its
    // single ODS. Rather than re-derive the prefix, build the canonical
    // single-ODS set and copy everything up to (but not including) its
    // ODS segment, which is the only segment we replace.
    let canonical = build_demo_display_set(canvas, object, position, palette, pixels);
    let mut out = Vec::new();
    let mut cur = 0;
    while cur < canonical.len() {
        let (seg, next) = read_segment(&canonical, cur).expect("demo set is well-formed");
        if seg.seg_type == SEG_ODS || seg.seg_type == SEG_END {
            break;
        }
        out.extend_from_slice(&canonical[cur..next]);
        cur = next;
    }

    // Build the full object payload: object_data_length (u24) + width +
    // height + RLE — identical to the single-segment ODS body minus the
    // per-fragment object_id / version / sequence-flag prefix.
    let rle = demo_rle(object, pixels);
    let obj_data_len = (rle.len() + 4) as u32; // width + height (4) + rle
    let mut payload = Vec::new();
    payload.push(((obj_data_len >> 16) & 0xFF) as u8);
    payload.push(((obj_data_len >> 8) & 0xFF) as u8);
    payload.push((obj_data_len & 0xFF) as u8);
    payload.extend_from_slice(&object.0.to_be_bytes());
    payload.extend_from_slice(&object.1.to_be_bytes());
    payload.extend_from_slice(&rle);

    // Split `payload` into `fragments` near-equal chunks. Each ODS body
    // is: object_id (2) + version (1) + sequence_flag (1) + chunk.
    let chunk_len = payload.len().div_ceil(fragments).max(1);
    let mut offset = 0;
    let mut idx = 0;
    while offset < payload.len() || (idx == 0 && payload.is_empty()) {
        let end = (offset + chunk_len).min(payload.len());
        let chunk = &payload[offset..end];
        let is_first = idx == 0;
        let is_last = end >= payload.len();
        let mut flag = 0u8;
        if is_first {
            flag |= 0x80;
        }
        if is_last {
            flag |= 0x40;
        }
        let mut ods = Vec::new();
        ods.extend_from_slice(&1u16.to_be_bytes()); // object id
        ods.push(0); // object version
        ods.push(flag);
        ods.extend_from_slice(chunk);
        segment(&mut out, 0, SEG_ODS, &ods);
        offset = end;
        idx += 1;
        if is_last {
            break;
        }
    }

    // END.
    segment(&mut out, 0, SEG_END, &[]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rle_round_trip_simple() {
        // A 4×1 bitmap: 1 1 0 0 — encoded as two one-pixel runs of colour
        // 1 plus a 2-pixel run of colour 0, then end-of-line.
        let rle: &[u8] = &[0x01, 0x01, 0x00, 0x02, 0x00, 0x00];
        let px = decode_rle(rle, 4, 1).unwrap();
        assert_eq!(px, vec![1, 1, 0, 0]);
    }

    // ---- RLE property sweep ---------------------------------------------
    //
    // The PGS RLE format documented in the module header has narrow rules
    // for short-vs-long runs and a separate "colour 0" branch with three
    // length encodings. The encoder picks the cheapest form per run; the
    // decoder must accept whichever the encoder emitted. These sweeps
    // verify `encode_rle ∘ decode_rle == identity` over a wide spread of
    // indexed bitmaps without external corpora.

    /// Deterministic LCG (same constants the composite tests use). Lets
    /// the sweeps run in plain Rust with no proptest dep.
    fn lcg(state: &mut u64) -> u32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 32) as u32
    }

    #[test]
    fn rle_roundtrip_random_widths_and_palettes() {
        // Sweep random bitmaps of varying width / height / palette size.
        // For each, encode → decode → check equality.
        let mut st = 0x9e37_79b9_7f4a_7c15u64;
        let mut sampled = 0usize;
        for _ in 0..1_500 {
            let width = ((lcg(&mut st) % 31) + 1) as usize; // 1..=31
            let height = ((lcg(&mut st) % 17) + 1) as usize; // 1..=17
            let palette_size = ((lcg(&mut st) % 12) + 1) as u8; // 1..=12 distinct values incl. 0
            let mut pixels = vec![0u8; width * height];
            for px in &mut pixels {
                *px = (lcg(&mut st) as u8) % palette_size;
            }
            let rle = encode_rle(&pixels, width, height);
            let back = decode_rle(&rle, width, height).unwrap_or_else(|e| {
                panic!(
                    "decode failed on roundtrip w={width} h={height} palette={palette_size}: {e:?}"
                )
            });
            assert_eq!(
                back, pixels,
                "round-trip diverged at w={width} h={height} palette={palette_size}"
            );
            sampled += 1;
        }
        assert_eq!(sampled, 1_500);
    }

    #[test]
    fn rle_roundtrip_long_runs() {
        // Force long-run encoding paths (14-bit length form). A 600-wide
        // row of a single non-zero colour exercises the
        // `0xC0|hi LL CC` branch; the all-transparent row exercises the
        // `0x40|hi LL` branch.
        for colour in [0u8, 1u8, 42u8] {
            for width in [64usize, 100, 256, 600, 1024] {
                let height = 3usize;
                let pixels = vec![colour; width * height];
                let rle = encode_rle(&pixels, width, height);
                let back = decode_rle(&rle, width, height).unwrap();
                assert_eq!(
                    back, pixels,
                    "long-run roundtrip diverged at colour={colour} width={width}"
                );
            }
        }
    }

    #[test]
    fn rle_roundtrip_alternating_short_runs() {
        // Alternating colours mean every run is length 1 — the encoder
        // must emit a literal byte per pixel for colour != 0, and the
        // `0x00 0x01` 2-byte form for each colour-0 pixel.
        let width = 16usize;
        let height = 4usize;
        let mut pixels = vec![0u8; width * height];
        for (i, px) in pixels.iter_mut().enumerate() {
            *px = if i % 2 == 0 { 0 } else { 7 };
        }
        let rle = encode_rle(&pixels, width, height);
        let back = decode_rle(&rle, width, height).unwrap();
        assert_eq!(back, pixels);
    }

    #[test]
    fn rle_roundtrip_one_pixel_rows() {
        // A 1×N bitmap stresses the end-of-line marker between
        // single-pixel rows.
        for height in [1usize, 2, 5, 11] {
            for colour in [0u8, 1u8, 250u8] {
                let pixels = vec![colour; height];
                let rle = encode_rle(&pixels, 1, height);
                let back = decode_rle(&rle, 1, height).unwrap();
                assert_eq!(back, pixels, "1×{height} colour={colour}");
            }
        }
    }

    #[test]
    fn rle_roundtrip_mixed_run_lengths_in_one_row() {
        // A handcrafted row that hits every encoder branch in sequence:
        // 1 literal, 3-singletons, 5-colour-run, 70-colour-run, then
        // 80-zero-run, then 200-zero-run.
        let mut pixels: Vec<u8> = Vec::new();
        pixels.push(9); // 1 literal
        pixels.extend(std::iter::repeat(5).take(3)); // 3 of colour 5 → 3 literals
        pixels.extend(std::iter::repeat(6).take(5)); // 5 of colour 6 → short 3-byte run
        pixels.extend(std::iter::repeat(7).take(70)); // 70 of colour 7 → 14-bit run
        pixels.extend(std::iter::repeat(0).take(80)); // 80 of colour 0 → 14-bit zero run
        pixels.extend(std::iter::repeat(0).take(200)); // a second zero block
        let width = pixels.len();
        let height = 1usize;
        let rle = encode_rle(&pixels, width, height);
        let back = decode_rle(&rle, width, height).unwrap();
        assert_eq!(back, pixels);
    }

    #[test]
    fn rle_size_shrinks_for_long_uniform_runs() {
        // A long, uniform run must shrink in the encoder — that's the
        // whole point of the format. The 14-bit-length form caps a run
        // at 4 bytes including its 2-byte end-of-line, so a 1000-wide
        // single-colour row should land well below the singleton bound.
        let width = 1000usize;
        let pixels = vec![3u8; width];
        let rle = encode_rle(&pixels, width, 1);
        assert!(
            rle.len() < 10,
            "long-run encoding too verbose: {} bytes for 1000-wide flat row",
            rle.len()
        );
    }

    // ---- RLE negative-input sweep ---------------------------------------
    //
    // The decoder must not panic on malformed RLE — that input arrives
    // from PGS payloads off arbitrary disks / networks. These tests pass
    // pathological byte streams in and assert the result is either a
    // graceful `Error::invalid` or a clamped success (the documented
    // behaviour for runs that overshoot the row).

    #[test]
    fn rle_truncated_escape_returns_invalid_not_panic() {
        // Lone 0x00 with no follow-up byte → escape begun, stream ended.
        let err = decode_rle(&[0x00], 4, 1);
        assert!(err.is_err(), "truncated escape must error: {err:?}");
    }

    #[test]
    fn rle_truncated_14bit_length_returns_invalid_not_panic() {
        // 0x00 0x4X marks the 14-bit-length-only form, needs one more byte.
        let err = decode_rle(&[0x00, 0x40], 4, 1);
        assert!(err.is_err(), "truncated 14-bit length must error: {err:?}");
    }

    #[test]
    fn rle_truncated_short_colour_run_returns_invalid_not_panic() {
        // 0x00 0x8L marks the short-colour-run form, needs colour byte.
        let err = decode_rle(&[0x00, 0x82], 4, 1);
        assert!(
            err.is_err(),
            "truncated short colour run must error: {err:?}"
        );
    }

    #[test]
    fn rle_truncated_long_colour_run_returns_invalid_not_panic() {
        // 0x00 0xCX marks the 14-bit-colour-run form, needs LL CC bytes.
        let err = decode_rle(&[0x00, 0xC0], 4, 1);
        assert!(
            err.is_err(),
            "truncated 14-bit colour run must error: {err:?}"
        );
        let err = decode_rle(&[0x00, 0xC0, 0xFF], 4, 1);
        assert!(
            err.is_err(),
            "14-bit colour run missing colour byte must error: {err:?}"
        );
    }

    #[test]
    fn rle_overlong_run_clamps_to_row_without_panic() {
        // The documented decoder behaviour for an over-long run is to
        // clamp to the row end and continue. Build a 4-wide row with a
        // 14-bit run claiming 100 pixels of colour 5, then a normal
        // end-of-line. The 4 pixels of colour 5 must come out cleanly.
        let rle: &[u8] = &[0x00, 0xC0, 0x64, 0x05, 0x00, 0x00];
        let px = decode_rle(rle, 4, 1).unwrap();
        assert_eq!(px, vec![5, 5, 5, 5]);
    }

    #[test]
    fn rle_too_many_lines_returns_invalid_not_panic() {
        // 1-row bitmap fed three end-of-lines: the second EOL pushes
        // `row` past `height` and must error rather than scribble.
        let rle: &[u8] = &[0x00, 0x00, 0x00, 0x00];
        let err = decode_rle(rle, 1, 1);
        assert!(err.is_err(), "extra EOL must error: {err:?}");
    }

    #[test]
    fn rle_pixel_past_end_returns_invalid_not_panic() {
        // A single literal pixel after the final end-of-line means
        // `row >= height` and the decoder hits the post-EOL pixel branch.
        let rle: &[u8] = &[0x01, 0x00, 0x00, 0x02];
        let err = decode_rle(rle, 1, 1);
        assert!(err.is_err(), "pixel past EOL must error: {err:?}");
    }

    #[test]
    fn rle_random_garbage_never_panics() {
        // Feed pseudo-random byte streams to the decoder over a sweep
        // of widths and heights. The result may be Ok or Err but must
        // not panic / corrupt memory / take time disproportionate to
        // the input length.
        let mut st = 0xfeed_face_dead_beefu64;
        for _ in 0..400 {
            let len = (lcg(&mut st) % 128) as usize;
            let mut bytes = vec![0u8; len];
            for b in &mut bytes {
                *b = lcg(&mut st) as u8;
            }
            let width = ((lcg(&mut st) % 17) + 1) as usize;
            let height = ((lcg(&mut st) % 9) + 1) as usize;
            let _ = decode_rle(&bytes, width, height); // just must not panic
        }
    }

    #[test]
    fn rle_roundtrip_all_one_colour_no_zero() {
        // Every pixel is a single non-zero colour. The encoder must
        // produce a single 14-bit-colour-run plus end-of-line.
        let width = 200usize;
        let height = 2usize;
        let pixels = vec![42u8; width * height];
        let rle = encode_rle(&pixels, width, height);
        let back = decode_rle(&rle, width, height).unwrap();
        assert_eq!(back, pixels);
        // Two rows × (one 4-byte 14-bit-run + 2-byte EOL) = 12 bytes is
        // the optimal encoding; we accept up to 16 bytes to allow for
        // any small encoder rearrangement.
        assert!(
            rle.len() <= 16,
            "uniform-row encoding too verbose: {} bytes",
            rle.len()
        );
    }

    #[test]
    fn rle_roundtrip_all_transparent_no_colour() {
        // Every pixel is colour 0. Each row encodes as one transparent
        // run + EOL.
        let width = 300usize;
        let height = 4usize;
        let pixels = vec![0u8; width * height];
        let rle = encode_rle(&pixels, width, height);
        let back = decode_rle(&rle, width, height).unwrap();
        assert_eq!(back, pixels);
        // Per row: 3-byte 14-bit zero run + 2-byte EOL = 5 bytes. ×4 = 20.
        assert!(
            rle.len() <= 28,
            "transparent-row encoding too verbose: {} bytes",
            rle.len()
        );
    }

    #[test]
    fn decodes_tiny_display_set() {
        // 2×2 bitmap: red red / green blue.
        let pixels = [1u8, 1, 2, 3];
        let palette = [
            (0u8, [0u8, 0, 0, 0]), // index 0 → transparent
            (1u8, [255u8, 0, 0, 255]),
            (2u8, [0u8, 255, 0, 255]),
            (3u8, [0u8, 0, 255, 255]),
        ];
        let blob = build_demo_display_set((2, 2), (2, 2), (0, 0), &palette, &pixels);

        let mut dec = make_decoder(&CodecParameters::video(CodecId::new(PGS_CODEC_ID))).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 90_000), blob).with_pts(0);
        dec.send_packet(&pkt).unwrap();
        let frame = dec.receive_frame().unwrap();
        let Frame::Video(v) = frame else {
            panic!("expected video frame");
        };
        assert_eq!(v.planes[0].stride, 2 * 4);
        let data = &v.planes[0].data;
        assert_eq!(data.len(), 2 * 2 * 4);
        // Row 0: red, red — the YCbCr round-trip is approximate, so
        // check R dominance + opaque alpha rather than exact bytes.
        let r0c0 = &data[0..4];
        let r0c1 = &data[4..8];
        assert!(
            r0c0[0] > 200 && r0c0[3] == 255,
            "not red-dominant: {:?}",
            r0c0
        );
        assert!(
            r0c1[0] > 200 && r0c1[3] == 255,
            "not red-dominant: {:?}",
            r0c1
        );
        // Row 1: green, blue — same dominance check.
        let g = &data[8..12];
        let b = &data[12..16];
        assert!(
            g[1] > g[0] && g[1] > g[2],
            "green pixel not dominant: {:?}",
            g
        );
        assert!(
            b[2] > b[0] && b[2] > b[1],
            "blue pixel not dominant: {:?}",
            b
        );
    }

    // ---- Fragmented-ODS reassembly --------------------------------------
    //
    // A real PGS muxer splits any object whose data overruns the
    // per-segment size limit across several ODS segments using the
    // first / continuation / last sequence-flag bits, and the decoder
    // must concatenate the fragments before interpreting the
    // object_data_length / width / height header or the RLE. The demo
    // encoder only ever emits a single first+last ODS, so without these
    // tests the `parse_ods_into` reassembly branch is unexercised.

    /// Decode a complete `.sup` display-set blob through the public
    /// decoder and return the single RGBA frame's pixel bytes.
    fn decode_one(blob: Vec<u8>) -> Vec<u8> {
        let mut dec = make_decoder(&CodecParameters::video(CodecId::new(PGS_CODEC_ID))).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 90_000), blob).with_pts(0);
        dec.send_packet(&pkt).unwrap();
        let Frame::Video(v) = dec.receive_frame().unwrap() else {
            panic!("expected video frame");
        };
        v.planes[0].data.clone()
    }

    #[test]
    fn fragmented_ods_matches_single_segment() {
        // Same object, encoded as 1 ODS vs. 2..=7 ODS fragments, must
        // decode to byte-identical RGBA. The split points fall at even
        // boundaries of (header ++ rle), so several of these land inside
        // the 7-byte width/height header — proving the decoder defers
        // interpretation until the whole object is reassembled.
        let pixels = [1u8, 1, 2, 3, 0, 1, 2, 3, 3, 2, 1, 0]; // 4×3
        let palette = [
            (0u8, [0u8, 0, 0, 0]),
            (1u8, [255u8, 0, 0, 255]),
            (2u8, [0u8, 255, 0, 255]),
            (3u8, [0u8, 0, 255, 255]),
        ];
        let canvas = (4u16, 3u16);
        let object = (4u16, 3u16);

        let single = decode_one(build_demo_display_set(
            canvas,
            object,
            (0, 0),
            &palette,
            &pixels,
        ));

        for n in 2..=7usize {
            let multi = decode_one(build_demo_display_set_fragmented(
                canvas,
                object,
                (0, 0),
                &palette,
                &pixels,
                n,
            ));
            assert_eq!(
                multi, single,
                "fragmented ODS ({n} segments) decoded differently from single-segment"
            );
        }
    }

    #[test]
    fn fragmented_ods_split_inside_header() {
        // A tiny 1×1 object whose total payload (7-byte header + a couple
        // of RLE bytes) is split into more fragments than it has bytes:
        // every fragment is at most one byte, so the very first boundary
        // is inside the object_data_length field. Reassembly must still
        // produce the correct single opaque pixel.
        let pixels = [1u8];
        let palette = [(0u8, [0u8, 0, 0, 0]), (1u8, [10u8, 200, 30, 255])];
        let blob = build_demo_display_set_fragmented((1, 1), (1, 1), (0, 0), &palette, &pixels, 32);
        let data = decode_one(blob);
        assert_eq!(data.len(), 4);
        // Green-dominant, opaque.
        assert!(
            data[1] > data[0] && data[1] > data[2] && data[3] == 255,
            "expected opaque green-dominant pixel, got {:?}",
            data
        );
    }

    #[test]
    fn incomplete_object_without_last_fragment_is_dropped_not_rendered() {
        // A first-but-never-last ODS leaves the object incomplete. The
        // decoder must not render a partial object; it emits the
        // canvas-sized frame with that composition object absent
        // (fully transparent here, since it's the only object).
        let pixels = [1u8, 1, 1, 1]; // 2×2
        let palette = [(0u8, [0u8, 0, 0, 0]), (1u8, [200u8, 0, 0, 255])];
        // Build the canonical set, then drop the END-terminating `last`
        // flag from its ODS by re-emitting only a `first` fragment.
        let frag2 = build_demo_display_set_fragmented((2, 2), (2, 2), (0, 0), &palette, &pixels, 2);
        // Keep everything up to and including the FIRST ODS fragment, then
        // splice in an END — so the decoder sees a first-without-last.
        let mut truncated = Vec::new();
        let mut cur = 0;
        let mut seen_first_ods = false;
        while cur < frag2.len() {
            let (seg, next) = read_segment(&frag2, cur).unwrap();
            if seg.seg_type == SEG_END {
                break;
            }
            truncated.extend_from_slice(&frag2[cur..next]);
            cur = next;
            if seg.seg_type == SEG_ODS {
                seen_first_ods = true;
                break;
            }
        }
        assert!(seen_first_ods, "expected to capture the first ODS fragment");
        // Terminating END (13-byte header, empty body).
        truncated.extend_from_slice(b"PG");
        truncated.extend_from_slice(&0u32.to_be_bytes());
        truncated.extend_from_slice(&0u32.to_be_bytes());
        truncated.push(SEG_END);
        truncated.extend_from_slice(&0u16.to_be_bytes());

        let data = decode_one(truncated);
        assert_eq!(data.len(), 2 * 2 * 4);
        for chunk in data.chunks(4) {
            assert_eq!(
                chunk,
                &[0, 0, 0, 0],
                "incomplete object must not paint any pixel"
            );
        }
    }

    // ---- PCS crop-rectangle handling ------------------------------------
    //
    // A Composition Segment may carry `object_cropped_flag = 1` followed by
    // an 8-byte cropping rectangle (`x`, `y`, `w`, `h`, each big-endian
    // `u16`) that selects a sub-region of the referenced Graphics Object
    // before it lands on the canvas. The whitepaper figure (Cropping stage
    // between Bitmap Object and Palette → Display Image) is the only place
    // this is illustrated in the BDA Part 3 material; field-order is the
    // same `(x, y, w, h)` shape the WDS window record uses elsewhere in
    // the same stream.

    /// Build a one-object display-set whose Composition Segment carries an
    /// `object_cropped_flag` + an 8-byte crop rectangle. Otherwise
    /// identical to [`build_demo_display_set`].
    fn build_cropped_display_set(
        canvas: (u16, u16),
        object: (u16, u16),
        position: (u16, u16),
        crop: (u16, u16, u16, u16),
        palette: &[(u8, [u8; 4])],
        pixels: &[u8],
    ) -> Vec<u8> {
        // Start from the canonical (uncropped) blob and rewrite only its
        // PCS segment to add the cropped flag + crop rectangle. Every
        // other segment (WDS / PDS / ODS / END) is reused unchanged.
        let canonical = build_demo_display_set(canvas, object, position, palette, pixels);
        let mut out = Vec::new();
        let mut cur = 0;
        let mut rewrote_pcs = false;
        while cur < canonical.len() {
            let (seg, next) = read_segment(&canonical, cur).expect("demo blob is well-formed");
            if seg.seg_type == SEG_PCS && !rewrote_pcs {
                rewrote_pcs = true;
                let mut pcs = Vec::new();
                pcs.extend_from_slice(&canvas.0.to_be_bytes());
                pcs.extend_from_slice(&canvas.1.to_be_bytes());
                pcs.push(0); // frame rate
                pcs.extend_from_slice(&1u16.to_be_bytes()); // composition number
                pcs.push(0); // composition state (normal)
                pcs.push(0); // palette update flag
                pcs.push(0); // palette id
                pcs.push(1); // one composition object
                pcs.extend_from_slice(&1u16.to_be_bytes()); // object id
                pcs.push(0); // window id
                pcs.push(0x80); // object_cropped_flag set
                pcs.extend_from_slice(&position.0.to_be_bytes());
                pcs.extend_from_slice(&position.1.to_be_bytes());
                pcs.extend_from_slice(&crop.0.to_be_bytes());
                pcs.extend_from_slice(&crop.1.to_be_bytes());
                pcs.extend_from_slice(&crop.2.to_be_bytes());
                pcs.extend_from_slice(&crop.3.to_be_bytes());
                // Re-emit segment header.
                out.extend_from_slice(b"PG");
                out.extend_from_slice(&seg.pts_90k.to_be_bytes());
                out.extend_from_slice(&seg.dts_90k.to_be_bytes());
                out.push(SEG_PCS);
                out.extend_from_slice(&(pcs.len() as u16).to_be_bytes());
                out.extend_from_slice(&pcs);
            } else {
                out.extend_from_slice(&canonical[cur..next]);
            }
            cur = next;
        }
        assert!(rewrote_pcs, "demo set must have a PCS to rewrite");
        out
    }

    #[test]
    fn crop_rect_parsed_into_composition_object() {
        // A PCS with object_cropped_flag set must populate `crop` on the
        // CompositionObject; the byte layout is `(x, y, w, h)` as four
        // big-endian u16s. Drive parse_pcs directly so we exercise the
        // parser in isolation of the renderer.
        let mut body = Vec::new();
        body.extend_from_slice(&100u16.to_be_bytes()); // width
        body.extend_from_slice(&50u16.to_be_bytes()); // height
        body.push(0); // frame rate
        body.extend_from_slice(&7u16.to_be_bytes()); // composition number
        body.push(0); // composition state
        body.push(0); // palette update flag
        body.push(0); // palette id
        body.push(1); // one composition object
        body.extend_from_slice(&3u16.to_be_bytes()); // object id
        body.push(0); // window id
        body.push(0x80); // cropped flag set
        body.extend_from_slice(&20u16.to_be_bytes()); // composition x
        body.extend_from_slice(&15u16.to_be_bytes()); // composition y
        body.extend_from_slice(&4u16.to_be_bytes()); // crop x
        body.extend_from_slice(&5u16.to_be_bytes()); // crop y
        body.extend_from_slice(&12u16.to_be_bytes()); // crop w
        body.extend_from_slice(&8u16.to_be_bytes()); // crop h

        let pcs = parse_pcs(&body).expect("parse_pcs should accept a valid cropped object");
        assert_eq!(pcs.objects.len(), 1);
        let co = &pcs.objects[0];
        assert!(co.cropped);
        assert_eq!(
            co.crop,
            Some(CropRect {
                x: 4,
                y: 5,
                w: 12,
                h: 8
            })
        );
    }

    #[test]
    fn crop_rect_zero_extent_rejected() {
        // A crop rectangle with `w == 0` or `h == 0` selects nothing —
        // authoring tools should use an erase display-set instead. The
        // parser rejects this as malformed so a bad encoder can't quietly
        // produce blank frames.
        for (cw, ch) in [(0u16, 8u16), (12u16, 0u16), (0u16, 0u16)] {
            let mut body = Vec::new();
            body.extend_from_slice(&100u16.to_be_bytes());
            body.extend_from_slice(&50u16.to_be_bytes());
            body.push(0);
            body.extend_from_slice(&1u16.to_be_bytes());
            body.push(0);
            body.push(0);
            body.push(0);
            body.push(1);
            body.extend_from_slice(&1u16.to_be_bytes());
            body.push(0);
            body.push(0x80);
            body.extend_from_slice(&0u16.to_be_bytes());
            body.extend_from_slice(&0u16.to_be_bytes());
            body.extend_from_slice(&0u16.to_be_bytes());
            body.extend_from_slice(&0u16.to_be_bytes());
            body.extend_from_slice(&cw.to_be_bytes());
            body.extend_from_slice(&ch.to_be_bytes());
            assert!(
                parse_pcs(&body).is_err(),
                "zero-extent crop ({cw}x{ch}) must be rejected"
            );
        }
    }

    #[test]
    fn crop_rect_truncated_is_rejected() {
        // PCS declares object_cropped_flag but the segment is one byte
        // short of carrying the eight-byte crop rectangle.
        let mut body = Vec::new();
        body.extend_from_slice(&100u16.to_be_bytes());
        body.extend_from_slice(&50u16.to_be_bytes());
        body.push(0);
        body.extend_from_slice(&1u16.to_be_bytes());
        body.push(0);
        body.push(0);
        body.push(0);
        body.push(1);
        body.extend_from_slice(&1u16.to_be_bytes());
        body.push(0);
        body.push(0x80);
        body.extend_from_slice(&0u16.to_be_bytes());
        body.extend_from_slice(&0u16.to_be_bytes());
        // Only seven of the eight crop bytes follow.
        body.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0]);
        assert!(parse_pcs(&body).is_err());
    }

    #[test]
    fn cropped_render_paints_only_sub_rectangle() {
        // 4×4 object: every pixel is colour 1 (opaque green). A crop of
        // (1, 1, 2, 2) selects a 2×2 sub-rect in object-space. With the
        // composition `(x, y) = (0, 0)` the cropped sub-rect lands at the
        // top-left corner of a 4×4 canvas; the rest of the canvas must
        // remain fully transparent because cropping discarded those rows
        // and columns of the source.
        let pixels = vec![1u8; 4 * 4];
        let palette = [(0u8, [0u8, 0, 0, 0]), (1u8, [40u8, 220, 40, 255])];
        let blob =
            build_cropped_display_set((4, 4), (4, 4), (0, 0), (1, 1, 2, 2), &palette, &pixels);
        let data = decode_one(blob);
        assert_eq!(data.len(), 4 * 4 * 4);
        for row in 0..4usize {
            for col in 0..4usize {
                let off = (row * 4 + col) * 4;
                let chunk = &data[off..off + 4];
                let inside = row < 2 && col < 2;
                if inside {
                    assert_eq!(
                        chunk[3], 255,
                        "cropped paint must reach pixel ({col},{row}): {chunk:?}"
                    );
                    assert!(
                        chunk[1] > chunk[0] && chunk[1] > chunk[2],
                        "expected green-dominant inside the crop, got {chunk:?} at ({col},{row})"
                    );
                } else {
                    assert_eq!(
                        chunk,
                        &[0u8, 0, 0, 0],
                        "pixel outside the crop window must be transparent at ({col},{row})"
                    );
                }
            }
        }
    }

    #[test]
    fn cropped_render_distinguishes_x_and_y_axes() {
        // A 4×3 object with two distinct colours per quadrant:
        //   row 0:  1 1 2 2
        //   row 1:  1 1 2 2
        //   row 2:  3 3 4 4
        // Cropping `(2, 0, 2, 2)` selects the top-right 2×2 (all colour 2);
        // cropping `(0, 2, 2, 1)` selects the bottom-left 2×1 (all
        // colour 3). The render output must agree.
        let pixels = vec![1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4];
        let palette = [
            (0u8, [0u8, 0, 0, 0]),
            (1u8, [10u8, 10, 10, 255]),
            (2u8, [255u8, 0, 0, 255]),
            (3u8, [0u8, 255, 0, 255]),
            (4u8, [0u8, 0, 255, 255]),
        ];

        // Top-right crop → red.
        let blob =
            build_cropped_display_set((4, 3), (4, 3), (0, 0), (2, 0, 2, 2), &palette, &pixels);
        let data = decode_one(blob);
        assert_eq!(data.len(), 4 * 3 * 4);
        for row in 0..3usize {
            for col in 0..4usize {
                let off = (row * 4 + col) * 4;
                let chunk = &data[off..off + 4];
                let inside = row < 2 && col < 2;
                if inside {
                    assert!(
                        chunk[0] > chunk[1] && chunk[0] > chunk[2] && chunk[3] == 255,
                        "expected red at ({col},{row}), got {chunk:?}"
                    );
                } else {
                    assert_eq!(chunk, &[0u8, 0, 0, 0]);
                }
            }
        }

        // Bottom-left crop → green, painted at canvas (0, 0) since the
        // composition position stays the same — the crop is on source
        // space, not destination.
        let blob =
            build_cropped_display_set((4, 3), (4, 3), (0, 0), (0, 2, 2, 1), &palette, &pixels);
        let data = decode_one(blob);
        for col in 0..2usize {
            let off = col * 4;
            let chunk = &data[off..off + 4];
            assert!(
                chunk[1] > chunk[0] && chunk[1] > chunk[2] && chunk[3] == 255,
                "expected green at top-left of canvas after vertical-y crop, got {chunk:?}"
            );
        }
        for col in 2..4usize {
            let off = col * 4;
            assert_eq!(&data[off..off + 4], &[0u8, 0, 0, 0]);
        }
    }

    #[test]
    fn crop_clipped_to_object_bounds() {
        // A 4×4 object cropped with a window that overflows the right
        // and bottom edges — e.g. `(2, 2, 10, 10)` — must reduce to the
        // intersection with the object's real bounds (here 2×2 starting
        // at (2, 2)). The render output is then the bottom-right 2×2
        // sub-image, placed at composition (0, 0).
        let pixels: Vec<u8> = (0..16u8).map(|i| if i >= 10 { 1 } else { 0 }).collect();
        // Object layout (one byte per cell):
        //   row 0: 0 0 0 0
        //   row 1: 0 0 0 0
        //   row 2: 1 1 1 1
        //   row 3: 1 1 1 1
        // After cropping `(2, 2, 10, 10)` we keep the 2×2 sub-rect at
        // object (2, 2) — four colour-1 cells.
        let palette = [(0u8, [0u8, 0, 0, 0]), (1u8, [80u8, 80, 200, 255])];
        let blob =
            build_cropped_display_set((4, 4), (4, 4), (0, 0), (2, 2, 10, 10), &palette, &pixels);
        let data = decode_one(blob);
        for row in 0..4usize {
            for col in 0..4usize {
                let off = (row * 4 + col) * 4;
                let chunk = &data[off..off + 4];
                if row < 2 && col < 2 {
                    assert_eq!(
                        chunk[3], 255,
                        "clipped 2×2 must paint top-left ({col},{row}): {chunk:?}"
                    );
                } else {
                    assert_eq!(
                        chunk,
                        &[0u8, 0, 0, 0],
                        "outside-window pixel ({col},{row}) must be transparent"
                    );
                }
            }
        }
    }

    #[test]
    fn crop_entirely_outside_object_paints_nothing() {
        // Crop starts at (10, 10) on a 4×4 object — there is no source
        // pixel to read after intersecting with object bounds. The
        // decoder must drop the composition silently and leave the canvas
        // fully transparent rather than panic or scribble.
        let pixels = vec![1u8; 4 * 4];
        let palette = [(0u8, [0u8, 0, 0, 0]), (1u8, [200u8, 0, 0, 255])];
        let blob =
            build_cropped_display_set((4, 4), (4, 4), (0, 0), (10, 10, 2, 2), &palette, &pixels);
        let data = decode_one(blob);
        assert_eq!(data.len(), 4 * 4 * 4);
        for chunk in data.chunks(4) {
            assert_eq!(chunk, &[0u8, 0, 0, 0]);
        }
    }

    // --- WDS typed parser + render-time window clipping ----------------

    #[test]
    fn wds_parses_typed_window_list() {
        // One window at (10, 20) sized 30×40 with id 7. The body layout
        // is `count` followed by `count` × 9-byte records.
        let body: Vec<u8> = vec![
            0x01, // 1 window
            0x07, // window_id
            0x00, 0x0A, // x = 10
            0x00, 0x14, // y = 20
            0x00, 0x1E, // w = 30
            0x00, 0x28, // h = 40
        ];
        let wins = parse_wds(&body).unwrap();
        assert_eq!(wins.len(), 1);
        assert_eq!(
            wins[0],
            WindowDefinition {
                window_id: 7,
                x: 10,
                y: 20,
                w: 30,
                h: 40
            }
        );
    }

    #[test]
    fn wds_zero_windows_yields_empty_list() {
        // The "erase" form WDS carries a single byte: count = 0. It is
        // valid and parses to an empty list (the renderer falls back to
        // "no clipping" in that case).
        let body: Vec<u8> = vec![0];
        let wins = parse_wds(&body).unwrap();
        assert!(wins.is_empty());
    }

    #[test]
    fn wds_rejects_truncated_body() {
        // Count says 1 window but only 8 of the 9 record bytes follow.
        let body: Vec<u8> = vec![0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        assert!(parse_wds(&body).is_err());
    }

    #[test]
    fn wds_rejects_trailing_bytes() {
        // Count says 0 but a stray byte follows — the body length must
        // match the declared count exactly so a malformed stream is
        // caught early rather than silently dropped.
        let body: Vec<u8> = vec![0x00, 0x00];
        assert!(parse_wds(&body).is_err());
    }

    #[test]
    fn wds_rejects_empty_body() {
        // A WDS body must always carry at least the count byte.
        assert!(parse_wds(&[]).is_err());
    }

    #[test]
    fn wds_rejects_zero_extent_window() {
        // A window with w = 0 or h = 0 is a malformed authoring tool —
        // not a deliberate "this window paints nothing" — and is
        // rejected at parse time.
        let body: Vec<u8> = vec![
            0x01, 0x00, // id
            0x00, 0x00, // x
            0x00, 0x00, // y
            0x00, 0x00, // w = 0
            0x00, 0x10, // h
        ];
        assert!(parse_wds(&body).is_err());
    }

    /// Build a single-display-set blob whose WDS declares an explicit
    /// window rectangle rather than the canonical full-canvas window
    /// emitted by [`build_demo_display_set`]. Used to drive the
    /// render-time window-clipping test below.
    fn build_demo_display_set_with_window(
        canvas: (u16, u16),
        object: (u16, u16),
        position: (u16, u16),
        window: (u16, u16, u16, u16),
        palette: &[(u8, [u8; 4])],
        pixels: &[u8],
    ) -> Vec<u8> {
        // Rewrite the WDS in the canonical blob to carry `window` rather
        // than the full-canvas default. Every other segment is left
        // unchanged so the only behavioural difference is the window
        // table the decoder records.
        let canonical = build_demo_display_set(canvas, object, position, palette, pixels);
        let mut out = Vec::new();
        let mut cur = 0;
        let mut rewrote_wds = false;
        while cur < canonical.len() {
            let (seg, next) = read_segment(&canonical, cur).expect("demo blob is well-formed");
            if seg.seg_type == SEG_WDS && !rewrote_wds {
                rewrote_wds = true;
                let mut wds = Vec::new();
                wds.push(1); // one window
                wds.push(0); // window id
                wds.extend_from_slice(&window.0.to_be_bytes());
                wds.extend_from_slice(&window.1.to_be_bytes());
                wds.extend_from_slice(&window.2.to_be_bytes());
                wds.extend_from_slice(&window.3.to_be_bytes());
                out.extend_from_slice(b"PG");
                out.extend_from_slice(&seg.pts_90k.to_be_bytes());
                out.extend_from_slice(&seg.dts_90k.to_be_bytes());
                out.push(SEG_WDS);
                out.extend_from_slice(&(wds.len() as u16).to_be_bytes());
                out.extend_from_slice(&wds);
            } else {
                out.extend_from_slice(&canonical[cur..next]);
            }
            cur = next;
        }
        assert!(rewrote_wds, "demo set must have a WDS to rewrite");
        out
    }

    #[test]
    fn window_clip_drops_object_pixels_outside_window() {
        // 4×4 canvas, 4×4 fully-opaque object positioned at (0, 0). The
        // WDS declares a 2×2 window at (1, 1), so only the four pixels
        // at (1,1), (2,1), (1,2), (2,2) of the canvas should be painted.
        // Everything else stays transparent.
        let pixels = vec![1u8; 4 * 4];
        let palette = [(0u8, [0u8, 0, 0, 0]), (1u8, [200u8, 50, 25, 255])];
        let blob = build_demo_display_set_with_window(
            (4, 4),
            (4, 4),
            (0, 0),
            (1, 1, 2, 2),
            &palette,
            &pixels,
        );
        let data = decode_one(blob);
        assert_eq!(data.len(), 4 * 4 * 4);
        for row in 0..4usize {
            for col in 0..4usize {
                let chunk = &data[(row * 4 + col) * 4..(row * 4 + col) * 4 + 4];
                let inside = (1..=2).contains(&row) && (1..=2).contains(&col);
                if inside {
                    assert_eq!(
                        chunk[3], 255,
                        "pixel inside the window must be opaque at ({col},{row})"
                    );
                } else {
                    assert_eq!(
                        chunk,
                        &[0u8, 0, 0, 0],
                        "pixel outside the window must be fully transparent at ({col},{row})"
                    );
                }
            }
        }
    }

    #[test]
    fn window_clip_drops_object_landing_entirely_outside() {
        // 8×8 canvas, 2×2 object at (5, 5), WDS window at (0, 0) sized
        // 4×4. The object lands wholly outside the window and the canvas
        // must remain fully transparent.
        let pixels = vec![1u8; 2 * 2];
        let palette = [(0u8, [0u8, 0, 0, 0]), (1u8, [200u8, 50, 25, 255])];
        let blob = build_demo_display_set_with_window(
            (8, 8),
            (2, 2),
            (5, 5),
            (0, 0, 4, 4),
            &palette,
            &pixels,
        );
        let data = decode_one(blob);
        assert_eq!(data.len(), 8 * 8 * 4);
        for chunk in data.chunks(4) {
            assert_eq!(chunk, &[0u8, 0, 0, 0]);
        }
    }

    // --- PCS composition_state surfacing -------------------------------

    /// Build a minimal PCS body with explicit `composition_state` and
    /// `palette_update_flag` bytes, zero composition objects, otherwise
    /// the canonical canvas / composition-number layout. Used by the
    /// composition-state tests below.
    fn build_pcs_body(
        canvas: (u16, u16),
        composition_number: u16,
        composition_state: u8,
        palette_update_flag: bool,
        palette_id: u8,
    ) -> Vec<u8> {
        let mut pcs = Vec::new();
        pcs.extend_from_slice(&canvas.0.to_be_bytes());
        pcs.extend_from_slice(&canvas.1.to_be_bytes());
        pcs.push(0); // frame rate
        pcs.extend_from_slice(&composition_number.to_be_bytes());
        pcs.push(composition_state);
        pcs.push(if palette_update_flag { 0x80 } else { 0x00 });
        pcs.push(palette_id);
        pcs.push(0); // zero composition objects
        pcs
    }

    #[test]
    fn pcs_normal_state_parses_into_surface_fields() {
        // Normal Case (0x00) is the default Composition-Number-only update
        // shape; nothing on the random-access surface should fire.
        let body = build_pcs_body((1920, 1080), 7, COMP_STATE_NORMAL, false, 0);
        let pcs = parse_pcs(&body).unwrap();
        assert_eq!(pcs.composition_state, COMP_STATE_NORMAL);
        assert!(!pcs.palette_update_flag);
        assert_eq!(pcs.palette_id, 0);
        assert!(
            !is_random_access(pcs.composition_state),
            "Normal Case must not be reported as a random-access point"
        );
    }

    #[test]
    fn pcs_acquisition_point_is_random_access() {
        // Acquisition Point (0x40) carries the full state needed to
        // resume rendering and is a valid seek target.
        let body = build_pcs_body((720, 480), 1, COMP_STATE_ACQUISITION, false, 0);
        let pcs = parse_pcs(&body).unwrap();
        assert_eq!(pcs.composition_state, COMP_STATE_ACQUISITION);
        assert!(is_random_access(pcs.composition_state));
    }

    #[test]
    fn pcs_epoch_start_is_random_access() {
        // Epoch Start (0x80) begins a brand-new epoch — the strongest
        // form of random-access point.
        let body = build_pcs_body((1280, 720), 1, COMP_STATE_EPOCH_START, false, 0);
        let pcs = parse_pcs(&body).unwrap();
        assert_eq!(pcs.composition_state, COMP_STATE_EPOCH_START);
        assert!(is_random_access(pcs.composition_state));
    }

    #[test]
    fn pcs_palette_update_flag_and_id_round_trip() {
        // The palette-update bit lives in the top bit of body[8]; the
        // palette_id byte lives at body[9]. Both should round-trip through
        // the parser unchanged.
        let body = build_pcs_body((1920, 1080), 3, COMP_STATE_NORMAL, true, 42);
        let pcs = parse_pcs(&body).unwrap();
        assert!(
            pcs.palette_update_flag,
            "top-bit-set palette_update_flag must surface as true"
        );
        assert_eq!(pcs.palette_id, 42);
    }

    #[test]
    fn pcs_palette_update_flag_lower_bits_are_ignored() {
        // The lower 7 bits of body[8] are reserved per the field layout —
        // they must not flip `palette_update_flag` on/off. Build a PCS
        // body with the reserved bits set but the top bit clear and
        // assert the parser still reports the flag as false.
        let mut body = build_pcs_body((1920, 1080), 3, COMP_STATE_NORMAL, false, 0);
        body[8] = 0x7F; // reserved bits set, top bit clear
        let pcs = parse_pcs(&body).unwrap();
        assert!(
            !pcs.palette_update_flag,
            "lower-bit noise in body[8] must not enable palette_update_flag"
        );
    }

    #[test]
    fn pcs_unknown_composition_state_passes_through_and_is_not_random_access() {
        // The HDMV spec defines three composition_state values; anything
        // else should pass through without being misclassified as a
        // random-access point. The parser surfaces it for downstream
        // consumers to act on as they see fit.
        let body = build_pcs_body((720, 480), 1, 0xAB, false, 0);
        let pcs = parse_pcs(&body).unwrap();
        assert_eq!(pcs.composition_state, 0xAB);
        assert!(!is_random_access(pcs.composition_state));
    }

    // --- Demuxer keyframe wiring ---------------------------------------

    /// Build a tiny `.sup` byte stream containing N display-sets with
    /// the given composition_state bytes, each ending with an END
    /// segment. Each set is a PCS with zero composition objects (no
    /// PDS / WDS / ODS needed) — the demuxer only cares about
    /// segmentation + PCS classification when assembling packets.
    fn build_sup_stream(states: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        for (i, &state) in states.iter().enumerate() {
            let pts = (i as u32 + 1) * 9_000; // 100 ms steps at 90 kHz
            let body = build_pcs_body((1920, 1080), i as u16, state, false, 0);
            push_segment(&mut out, pts, SEG_PCS, &body);
            push_segment(&mut out, pts, SEG_END, &[]);
        }
        out
    }

    #[test]
    fn demuxer_marks_only_random_access_packets_as_keyframes() {
        // Build a stream that interleaves all three composition_state
        // classes: Epoch Start, Normal Case, Acquisition Point, Normal
        // Case, Normal Case. The demuxer must mark packets 0 and 2 as
        // keyframes (random-access entry points) and leave the others
        // un-flagged.
        let states = [
            COMP_STATE_EPOCH_START,
            COMP_STATE_NORMAL,
            COMP_STATE_ACQUISITION,
            COMP_STATE_NORMAL,
            COMP_STATE_NORMAL,
        ];
        let stream = build_sup_stream(&states);
        let resolver = oxideav_core::NullCodecResolver;
        let input: Box<dyn ReadSeek> = Box::new(std::io::Cursor::new(stream));
        let mut dmx = open_pgs(input, &resolver).unwrap();
        let mut keyframe_flags = Vec::new();
        loop {
            match dmx.next_packet() {
                Ok(pkt) => keyframe_flags.push(pkt.flags.keyframe),
                Err(Error::Eof) => break,
                Err(e) => panic!("unexpected demux error: {e:?}"),
            }
        }
        assert_eq!(
            keyframe_flags,
            vec![true, false, true, false, false],
            "expected only Epoch Start + Acquisition Point packets to be keyframes",
        );
    }

    #[test]
    fn demuxer_clears_random_access_state_between_packets() {
        // A random-access display-set followed by a Normal Case set must
        // see the second packet's `keyframe` flag drop back to false —
        // i.e. the demuxer must not carry the classification across
        // display-set boundaries. The regression this guards against is
        // a stuck flag that would mark every set after the first
        // Acquisition Point as a keyframe forever.
        let states = [COMP_STATE_ACQUISITION, COMP_STATE_NORMAL];
        let stream = build_sup_stream(&states);
        let resolver = oxideav_core::NullCodecResolver;
        let input: Box<dyn ReadSeek> = Box::new(std::io::Cursor::new(stream));
        let mut dmx = open_pgs(input, &resolver).unwrap();
        let p0 = dmx.next_packet().unwrap();
        let p1 = dmx.next_packet().unwrap();
        assert!(p0.flags.keyframe, "first set is Acquisition Point");
        assert!(!p1.flags.keyframe, "second set is Normal Case");
    }

    #[test]
    fn demuxer_unknown_composition_state_is_not_a_keyframe() {
        // An out-of-range composition_state byte must not be promoted to
        // a keyframe. The demuxer treats `is_random_access` as the only
        // promotion gate, so an unknown classification means "not a safe
        // seek target."
        let states = [0xCDu8];
        let stream = build_sup_stream(&states);
        let resolver = oxideav_core::NullCodecResolver;
        let input: Box<dyn ReadSeek> = Box::new(std::io::Cursor::new(stream));
        let mut dmx = open_pgs(input, &resolver).unwrap();
        let pkt = dmx.next_packet().unwrap();
        assert!(
            !pkt.flags.keyframe,
            "unknown composition_state must not promote a packet to a keyframe"
        );
    }

    #[test]
    fn encoder_emits_epoch_start_so_decoded_packets_are_random_access() {
        // The encoder writes 0x80 (Epoch Start) for every emitted display
        // set — each `send_frame` call produces a packet that decodes
        // standalone, so it must be classified as a random-access point.
        // Verify by feeding the encoder output back through the parser
        // path: each set must report Epoch Start.
        let pixels: Vec<u8> = (0..(4 * 3 * 4))
            .map(|i| if i % 4 == 3 { 255 } else { 200 })
            .collect();
        let frame = Frame::Video(VideoFrame {
            pts: Some(0),
            planes: vec![VideoPlane {
                stride: 4 * 4,
                data: pixels,
            }],
        });
        let params = CodecParameters::video(CodecId::new(PGS_CODEC_ID));
        let mut enc = make_encoder(&params).unwrap();
        enc.send_frame(&frame).unwrap();
        let packet = enc.receive_packet().unwrap();

        let (seg, _next) = read_segment(&packet.data, 0).unwrap();
        assert_eq!(seg.seg_type, SEG_PCS);
        let pcs = parse_pcs(&seg.body).unwrap();
        assert_eq!(
            pcs.composition_state, COMP_STATE_EPOCH_START,
            "encoder must emit Epoch Start on every set (each call is standalone)"
        );
        assert!(is_random_access(pcs.composition_state));
    }
}
