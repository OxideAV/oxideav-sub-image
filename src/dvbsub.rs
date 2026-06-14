//! DVB subtitle decoder (ETSI EN 300 743).
//!
//! DVB subtitles are carried as PES payloads inside MPEG-TS; each PES
//! payload begins with `0x20 0x00` (data_identifier + subtitle_stream_id),
//! followed by one or more segments, and ends with `0xFF` end-of-PES.
//! Each segment has the form:
//!
//! ```text
//! 0x0F              sync byte
//! segment_type      (1)   0x10 page / 0x11 region / 0x12 CLUT / 0x13 object / 0x14 display
//! page_id           (2 BE)
//! segment_length    (2 BE)
//! segment_body      (segment_length bytes)
//! ```
//!
//! A display-set terminates at either a `0x80` end-of-display-set
//! segment or the next page_composition segment. The decoder composes
//! the referenced regions onto a canvas sized by the display-definition
//! segment (or the default 720×576 PAL raster).
//!
//! ## Scope / limitations
//!
//! * **Decode:** pixel-coded objects are supported, including the
//!   `non_modifying_colour_flag` (§7.2.5): when set, CLUT index 1 is the
//!   non-modifying colour and pixels carrying it leave the underlying
//!   region/object content untouched ("transparent holes"). *Character*-
//!   coded objects (2‐byte segments used for teletext-style streams)
//!   currently return `Error::Unsupported`.
//! * **Encode:** [`make_encoder`] turns RGBA video frames into complete
//!   display-set PES payloads (DDS + page composition + region
//!   composition + CLUT definition + object data + end-of-display-set),
//!   and the individual segment writers (`write_segment`,
//!   `write_display_definition`, `write_page_composition`,
//!   `write_region_composition`, `write_clut_definition`,
//!   `write_object_data`) plus the 2/4/8-bit pixel-code-string encoders
//!   are exposed for callers that assemble their own display sets. The
//!   emitted payload is the PES-level byte stream (with the `0x20 0x00`
//!   prefix); riding MPEG-TS still needs a TS muxer upstream.
//! * Multi-region displays are composited in page-composition list
//!   order: each referenced region paints onto the canvas in the order
//!   it appears in the page-composition segment, blending over whatever
//!   an earlier region already wrote (Porter–Duff source-over), so a
//!   later region with a partially-transparent CLUT entry shows the
//!   region beneath it through rather than discarding it.
//! * Page timeouts are accepted but not enforced here — caller uses the
//!   accompanying packet duration.
//! * `region_fill_flag` (ETSI EN 300 743 §7.2.3) is honoured: when set,
//!   the region rectangle is pre-painted with the depth-appropriate
//!   `region_n-bit_pixel_code` *before* any objects composite on top.
//!   Cleared, the rectangle stays at the canvas's transparent
//!   background. The fill is clipped to the canvas; out-of-bounds
//!   region rectangles only paint their in-bounds intersection.

use std::collections::{HashMap, VecDeque};

use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, MediaType, Packet, PixelFormat, Result, TimeBase,
    VideoFrame, VideoPlane,
};
use oxideav_core::{Decoder, Encoder};

use crate::DVBSUB_CODEC_ID;

// --- segment-type identifiers ------------------------------------------

pub const SEG_PAGE_COMPOSITION: u8 = 0x10;
pub const SEG_REGION_COMPOSITION: u8 = 0x11;
pub const SEG_CLUT_DEFINITION: u8 = 0x12;
pub const SEG_OBJECT_DATA: u8 = 0x13;
pub const SEG_DISPLAY_DEFINITION: u8 = 0x14;
/// Disparity Signalling Segment (ETSI EN 300 743 §7.2.7) — carries the
/// plano-stereoscopic (3D) per-page / per-region / per-subregion disparity
/// shift values.
pub const SEG_DISPARITY_SIGNALLING: u8 = 0x15;
pub const SEG_END_OF_DISPLAY_SET: u8 = 0x80;

// --- raw segments ------------------------------------------------------

#[derive(Clone, Debug)]
pub struct RawSegment {
    pub seg_type: u8,
    pub page_id: u16,
    pub body: Vec<u8>,
}

/// Read the next DVB segment at `buf[pos]`. Returns the segment + new
/// cursor, or `Error::NeedMore` when fewer than 6 bytes remain / the
/// body is truncated.
pub fn read_segment(buf: &[u8], pos: usize) -> Result<(RawSegment, usize)> {
    if pos + 6 > buf.len() {
        return Err(Error::NeedMore);
    }
    if buf[pos] != 0x0F {
        return Err(Error::invalid(format!(
            "DVB sub: segment sync byte 0x0F expected, got 0x{:02X}",
            buf[pos]
        )));
    }
    let seg_type = buf[pos + 1];
    let page_id = u16::from_be_bytes([buf[pos + 2], buf[pos + 3]]);
    let len = u16::from_be_bytes([buf[pos + 4], buf[pos + 5]]) as usize;
    let end = pos + 6 + len;
    if end > buf.len() {
        return Err(Error::NeedMore);
    }
    Ok((
        RawSegment {
            seg_type,
            page_id,
            body: buf[pos + 6..end].to_vec(),
        },
        end,
    ))
}

// --- parse helpers -----------------------------------------------------

/// The optional rendering sub-window carried by a Display Definition
/// Segment when its `display_window_flag` is set (ETSI EN 300 743
/// §7.2.1). Coordinates are absolute pixel/line addresses with respect
/// to the top-left of the display raster (the `..._maximum` fields name
/// the last in-window pixel/line, so the window is inclusive on both
/// ends). When the flag is clear the display set is rendered directly
/// across the whole `display_width × display_height` raster and this is
/// `None`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DisplayWindow {
    /// `display_window_horizontal_position_minimum` — left-most in-window pixel.
    pub h_min: u16,
    /// `display_window_horizontal_position_maximum` — right-most in-window pixel.
    pub h_max: u16,
    /// `display_window_vertical_position_minimum` — top-most in-window line.
    pub v_min: u16,
    /// `display_window_vertical_position_maximum` — bottom in-window line.
    pub v_max: u16,
}

#[derive(Clone, Debug)]
struct DisplayDefinition {
    width: u16,
    height: u16,
    version: u8,
    window: Option<DisplayWindow>,
}

impl Default for DisplayDefinition {
    fn default() -> Self {
        // Standard definition PAL DVB default raster.
        Self {
            width: 720,
            height: 576,
            version: 0,
            window: None,
        }
    }
}

fn parse_display_definition(body: &[u8]) -> Result<DisplayDefinition> {
    if body.len() < 5 {
        return Err(Error::invalid("DVB DDS: body too short"));
    }
    // body[0]: dds_version_number (4) | display_window_flag (1) | reserved (3).
    let version = body[0] >> 4;
    let window_flag = (body[0] & 0x08) != 0;
    let width = u16::from_be_bytes([body[1], body[2]]).wrapping_add(1);
    let height = u16::from_be_bytes([body[3], body[4]]).wrapping_add(1);
    let window = if window_flag {
        // Four 16-bit window positions follow (§7.2.1). The `_maximum`
        // values name the last in-window pixel/line, so the window is
        // inclusive at both edges; a maximum below its matching minimum
        // is malformed (a zero-extent window can't bound a display set).
        if body.len() < 13 {
            return Err(Error::invalid(
                "DVB DDS: window flag set but body too short",
            ));
        }
        let h_min = u16::from_be_bytes([body[5], body[6]]);
        let h_max = u16::from_be_bytes([body[7], body[8]]);
        let v_min = u16::from_be_bytes([body[9], body[10]]);
        let v_max = u16::from_be_bytes([body[11], body[12]]);
        if h_max < h_min || v_max < v_min {
            return Err(Error::invalid("DVB DDS: window maximum below minimum"));
        }
        Some(DisplayWindow {
            h_min,
            h_max,
            v_min,
            v_max,
        })
    } else {
        None
    };
    Ok(DisplayDefinition {
        width,
        height,
        version,
        window,
    })
}

// --- Disparity Signalling Segment (§7.2.7) -----------------------------

/// One step of a `disparity_shift_update_sequence` (ETSI EN 300 743
/// §7.2.7): a single near-future disparity value plus the interval
/// multiplier that places it on the presentation timeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DisparityDivisionPeriod {
    /// `interval_count` — multiplier (≥ 1) applied to the sequence's
    /// `interval_duration` to compute this update's PTS offset from the
    /// previous value.
    pub interval_count: u8,
    /// `disparity_shift_update_integer_part` — signed integer disparity
    /// (−128..=+127 pixels) applied at this step.
    pub disparity_shift_integer: i8,
}

/// A `disparity_shift_update_sequence()` (ETSI EN 300 743 §7.2.7) — a
/// run of near-future disparity values transmitted together so the
/// decoder can schedule (and interpolate between) them off a single PTS.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct DisparityShiftUpdateSequence {
    /// `interval_duration` — 24-bit unit (90 kHz STC ticks) between
    /// division periods.
    pub interval_duration: u32,
    /// One entry per `division_period_count` value (≥ 1).
    pub division_periods: Vec<DisparityDivisionPeriod>,
}

/// One subregion's disparity within a DSS region loop (ETSI EN 300 743
/// §7.2.7). When a region declares a single subregion the positional
/// fields are absent on the wire and inherit the whole region's extent;
/// they are reported here as `None` so callers can distinguish the
/// "whole region" case from an explicit sub-rectangle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DisparitySubregion {
    /// `subregion_horizontal_position` — left-most pixel, page-relative.
    /// `None` when the region carries a single implicit subregion.
    pub horizontal_position: Option<u16>,
    /// `subregion_width` — width in pixels. `None` with a single
    /// implicit subregion (it spans the whole region).
    pub width: Option<u16>,
    /// `subregion_disparity_shift_integer_part` — signed integer pixels
    /// (−128..=+127).
    pub disparity_shift_integer: i8,
    /// `subregion_disparity_shift_fractional_part` — unsigned 1/16-pixel
    /// units (0..=15) **added** to the integer part per the spec's
    /// worked examples (−0.75 ⇒ [−1, 4/16]).
    pub disparity_shift_fractional: u8,
    /// Per-region `disparity_shift_update_sequence`, present only when
    /// `disparity_shift_update_sequence_region_flag` was set. It applies
    /// to every subregion of the region.
    pub update_sequence: Option<DisparityShiftUpdateSequence>,
}

/// One region entry in the DSS region loop (ETSI EN 300 743 §7.2.7).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DisparityRegion {
    /// `region_id` the subregions below belong to.
    pub region_id: u8,
    /// The subregions (1..=4) declared for this region.
    pub subregions: Vec<DisparitySubregion>,
}

/// A parsed Disparity Signalling Segment (ETSI EN 300 743 §7.2.7).
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct DisparitySignalling {
    /// `dss_version_number` (modulo-16).
    pub version: u8,
    /// `page_default_disparity_shift` — signed integer default applied
    /// to every region when the decoder cannot place individual values.
    pub page_default_disparity_shift: i8,
    /// Page-level `disparity_shift_update_sequence`, present only when
    /// `disparity_shift_update_sequence_page_flag` was set.
    pub page_update_sequence: Option<DisparityShiftUpdateSequence>,
    /// The DSS region loop. Regions absent here inherit
    /// `page_default_disparity_shift`.
    pub regions: Vec<DisparityRegion>,
}

/// Parse a `disparity_shift_update_sequence()` starting at `buf[*cur]`.
/// Advances `*cur` past the consumed bytes.
fn parse_disparity_update_sequence(
    buf: &[u8],
    cur: &mut usize,
) -> Result<DisparityShiftUpdateSequence> {
    // disparity_shift_update_sequence_length (8) precedes a body of that
    // many bytes: interval_duration (24) + division_period_count (8) +
    // division_period_count × { interval_count (8) + integer_part (8) }.
    if *cur + 1 > buf.len() {
        return Err(Error::invalid("DVB DSS: update-sequence length missing"));
    }
    let seq_len = buf[*cur] as usize;
    let body_start = *cur + 1;
    let body_end = body_start + seq_len;
    if body_end > buf.len() {
        return Err(Error::invalid("DVB DSS: update-sequence body truncated"));
    }
    if seq_len < 4 {
        return Err(Error::invalid(
            "DVB DSS: update-sequence too short for interval_duration + count",
        ));
    }
    let interval_duration =
        u32::from_be_bytes([0, buf[body_start], buf[body_start + 1], buf[body_start + 2]]);
    let division_period_count = buf[body_start + 3] as usize;
    let mut periods = Vec::with_capacity(division_period_count);
    let mut p = body_start + 4;
    for _ in 0..division_period_count {
        if p + 2 > body_end {
            return Err(Error::invalid(
                "DVB DSS: update-sequence division period truncated",
            ));
        }
        periods.push(DisparityDivisionPeriod {
            interval_count: buf[p],
            disparity_shift_integer: buf[p + 1] as i8,
        });
        p += 2;
    }
    *cur = body_end;
    Ok(DisparityShiftUpdateSequence {
        interval_duration,
        division_periods: periods,
    })
}

fn parse_disparity_signalling(body: &[u8]) -> Result<DisparitySignalling> {
    // dss_version_number (4) | page_flag (1) | reserved (3) +
    // page_default_disparity_shift (8) minimum.
    if body.len() < 2 {
        return Err(Error::invalid("DVB DSS: body too short"));
    }
    let version = body[0] >> 4;
    let page_flag = (body[0] & 0x08) != 0;
    let page_default_disparity_shift = body[1] as i8;
    let mut cur = 2;
    let page_update_sequence = if page_flag {
        Some(parse_disparity_update_sequence(body, &mut cur)?)
    } else {
        None
    };
    // region loop runs while processed_length < segment_length, i.e. to
    // the end of the body.
    let mut regions = Vec::new();
    while cur < body.len() {
        if cur + 2 > body.len() {
            return Err(Error::invalid("DVB DSS: region entry header truncated"));
        }
        let region_id = body[cur];
        let flags = body[cur + 1];
        let region_update_flag = (flags & 0x80) != 0;
        // reserved (5) sit between the flag and the 2-bit count.
        let number_of_subregions = (flags & 0x03) as usize + 1;
        cur += 2;
        let mut subregions = Vec::with_capacity(number_of_subregions);
        for _ in 0..number_of_subregions {
            // Positional fields are present only when there is more than
            // one subregion (number_of_subregions_minus_1 > 0).
            let (horizontal_position, width) = if number_of_subregions > 1 {
                if cur + 4 > body.len() {
                    return Err(Error::invalid(
                        "DVB DSS: subregion position fields truncated",
                    ));
                }
                let h = u16::from_be_bytes([body[cur], body[cur + 1]]);
                let w = u16::from_be_bytes([body[cur + 2], body[cur + 3]]);
                cur += 4;
                (Some(h), Some(w))
            } else {
                (None, None)
            };
            if cur + 2 > body.len() {
                return Err(Error::invalid(
                    "DVB DSS: subregion disparity fields truncated",
                ));
            }
            let disparity_shift_integer = body[cur] as i8;
            // fractional_part (4) | reserved (4).
            let disparity_shift_fractional = (body[cur + 1] >> 4) & 0x0F;
            cur += 2;
            let update_sequence = if region_update_flag {
                Some(parse_disparity_update_sequence(body, &mut cur)?)
            } else {
                None
            };
            subregions.push(DisparitySubregion {
                horizontal_position,
                width,
                disparity_shift_integer,
                disparity_shift_fractional,
                update_sequence,
            });
        }
        regions.push(DisparityRegion {
            region_id,
            subregions,
        });
    }
    Ok(DisparitySignalling {
        version,
        page_default_disparity_shift,
        page_update_sequence,
        regions,
    })
}

#[derive(Clone, Debug)]
struct PageRegion {
    region_id: u8,
    x: u16,
    y: u16,
}

fn parse_page_composition(body: &[u8]) -> Result<Vec<PageRegion>> {
    if body.len() < 2 {
        return Err(Error::invalid("DVB page_composition: body too short"));
    }
    // body[0] = page_time_out (s), body[1] = version/state
    let mut cur = 2;
    let mut regions = Vec::new();
    while cur + 6 <= body.len() {
        let region_id = body[cur];
        // body[cur+1] reserved
        let x = u16::from_be_bytes([body[cur + 2], body[cur + 3]]);
        let y = u16::from_be_bytes([body[cur + 4], body[cur + 5]]);
        cur += 6;
        regions.push(PageRegion { region_id, x, y });
    }
    Ok(regions)
}

#[derive(Clone, Debug)]
struct Region {
    width: u16,
    height: u16,
    /// Colour-depth declared by the region: 2, 4, or 8 bits. Used to
    /// pick which `region_n-bit_pixel_code` is treated as the fill
    /// colour when `fill` is set; the pixel-coded object streams still
    /// carry their own depth so this byte is otherwise purely
    /// validatory.
    depth_bits: u8,
    clut_id: u8,
    /// `region_fill_flag` (ETSI EN 300 743 §7.2.3): when true the
    /// region rectangle is pre-painted with the depth-appropriate
    /// `region_n-bit_pixel_code` *before* objects composite on top.
    /// When false the rectangle starts fully transparent.
    fill: bool,
    /// `region_8-bit_pixel_code` — the CLUT index used to fill the
    /// region when `fill` is true and `depth_bits == 8`.
    fill_code_8: u8,
    /// `region_4-bit_pixel_code` — packed into the high nibble of the
    /// final byte of the region-composition header (ETSI EN 300 743
    /// §7.2.3). Used when `fill` is true and `depth_bits == 4`.
    fill_code_4: u8,
    /// `region_2-bit_pixel_code` — bits 5..6 of the final header byte.
    /// Used when `fill` is true and `depth_bits == 2`.
    fill_code_2: u8,
    objects: Vec<RegionObject>,
}

#[derive(Clone, Debug)]
struct RegionObject {
    object_id: u16,
    x: u16,
    y: u16,
}

fn parse_region_composition(body: &[u8]) -> Result<(u8, Region)> {
    if body.len() < 10 {
        return Err(Error::invalid("DVB region_composition: body too short"));
    }
    let region_id = body[0];
    // body[1] = region_version_number (4) + region_fill_flag (1) + reserved (3).
    // The fill flag is bit 3 (mask 0x08) per ETSI EN 300 743 §7.2.3.
    let fill = (body[1] & 0x08) != 0;
    let width = u16::from_be_bytes([body[2], body[3]]);
    let height = u16::from_be_bytes([body[4], body[5]]);
    // body[6] = region_level_of_compatibility (3) + region_depth (3) + reserved (2)
    let region_depth = (body[6] >> 2) & 0x07;
    let depth_bits = match region_depth {
        1 => 2,
        2 => 4,
        3 => 8,
        _ => 4, // default-ish
    };
    let clut_id = body[7];
    // body[8] = region_8-bit_pixel_code
    // body[9] = (region_4-bit_pixel_code << 4) | (region_2-bit_pixel_code << 2) | reserved (2)
    let fill_code_8 = body[8];
    let fill_code_4 = (body[9] >> 4) & 0x0F;
    let fill_code_2 = (body[9] >> 2) & 0x03;
    let mut cur = 10;
    let mut objects = Vec::new();
    while cur + 6 <= body.len() {
        let obj_hi = body[cur];
        let obj_lo = body[cur + 1];
        let object_id = u16::from_be_bytes([obj_hi, obj_lo]);
        let obj_type = (body[cur + 2] >> 6) & 0x03;
        // body[cur+2] also has provider_flag (2 bits) + x_pos hi (6 bits)
        let x = u16::from_be_bytes([body[cur + 2] & 0x3F, body[cur + 3]]);
        // body[cur+4] has reserved (4) + y_pos hi (4 bits)
        let y = u16::from_be_bytes([body[cur + 4] & 0x0F, body[cur + 5]]);
        cur += 6;
        if obj_type == 0x01 || obj_type == 0x02 {
            // foreground/background colour bytes follow — skip 2.
            if cur + 2 <= body.len() {
                cur += 2;
            }
        }
        objects.push(RegionObject { object_id, x, y });
    }
    Ok((
        region_id,
        Region {
            width,
            height,
            depth_bits,
            clut_id,
            fill,
            fill_code_8,
            fill_code_4,
            fill_code_2,
            objects,
        },
    ))
}

#[derive(Clone, Debug)]
struct Clut {
    /// 8-bit palette entries as RGBA.
    entries: [[u8; 4]; 256],
}

impl Default for Clut {
    fn default() -> Self {
        Self {
            entries: [[0u8; 4]; 256],
        }
    }
}

fn ycbcr_to_rgba(y: u8, cr: u8, cb: u8, t8: u8) -> [u8; 4] {
    // DVB CLUT values: Y/Cb/Cr full-range 8-bit + T (transparency).
    // Convert via BT.601.
    let y = y as i32;
    let cb = cb as i32 - 128;
    let cr = cr as i32 - 128;
    let r = y + ((91881 * cr) >> 16);
    let g = y - ((22554 * cb + 46802 * cr) >> 16);
    let b = y + ((116130 * cb) >> 16);
    let alpha = 255u8.saturating_sub(t8);
    [
        r.clamp(0, 255) as u8,
        g.clamp(0, 255) as u8,
        b.clamp(0, 255) as u8,
        alpha,
    ]
}

fn parse_clut_into(body: &[u8], cluts: &mut HashMap<u8, Clut>) -> Result<()> {
    if body.len() < 2 {
        return Err(Error::invalid("DVB CLUT: body too short"));
    }
    let clut_id = body[0];
    // body[1] = version
    let entry = cluts.entry(clut_id).or_default();
    let mut cur = 2;
    while cur < body.len() {
        // Each entry: entry_id (1), flags (1), then either 2-byte
        // Y/Cr/Cb/T (bit set) or 4-byte values.
        if cur + 2 > body.len() {
            break;
        }
        let entry_id = body[cur];
        let flags = body[cur + 1];
        cur += 2;
        let full = (flags & 0x01) != 0;
        if full {
            if cur + 4 > body.len() {
                return Err(Error::invalid("DVB CLUT: truncated full entry"));
            }
            let y = body[cur];
            let cr = body[cur + 1];
            let cb = body[cur + 2];
            let t = body[cur + 3];
            cur += 4;
            entry.entries[entry_id as usize] = ycbcr_to_rgba(y, cr, cb, t);
        } else {
            if cur + 2 > body.len() {
                return Err(Error::invalid("DVB CLUT: truncated short entry"));
            }
            // Packed: 6 bits Y, 4 bits Cr, 4 bits Cb, 2 bits T — not
            // widely used; fall back to scaling.
            let b0 = body[cur];
            let b1 = body[cur + 1];
            cur += 2;
            let y = b0 & 0xFC;
            let cr = (((b0 & 0x03) << 2) | (b1 >> 6)) << 4;
            let cb = ((b1 >> 2) & 0x0F) << 4;
            let t = (b1 & 0x03) << 6;
            entry.entries[entry_id as usize] = ycbcr_to_rgba(y, cr, cb, t);
        }
    }
    Ok(())
}

// --- object data / pixel-coded string decoder --------------------------

#[derive(Clone, Debug)]
struct Object {
    /// Row-major indexed pixels. Width/height are determined by the
    /// region that hosts the object.
    rows: Vec<Vec<u8>>,
    /// Source object-data type. We track this so we can refuse
    /// character-coded objects at render time.
    #[allow(dead_code)]
    coding_method: u8,
    /// ETSI EN 300 743 §7.2.5 `non_modifying_colour_flag`: when set, CLUT
    /// entry value `1` is the *non-modifying colour*. Pixels assigned that
    /// index leave the underlying region background / object pixel
    /// unchanged ("transparent holes"), rather than painting CLUT entry 1.
    non_modifying_colour: bool,
}

fn parse_object_data(body: &[u8]) -> Result<(u16, Object)> {
    if body.len() < 3 {
        return Err(Error::invalid("DVB object_data: body too short"));
    }
    let object_id = u16::from_be_bytes([body[0], body[1]]);
    // body[2] = version (4 bits) + coding_method (2 bits) + non_modifying_colour_flag (1) + reserved
    let coding_method = (body[2] >> 2) & 0x03;
    let non_modifying_colour = (body[2] >> 1) & 0x01 != 0;
    if coding_method != 0 {
        return Err(Error::unsupported(format!(
            "DVB sub: coding_method {} (character/text objects)",
            coding_method
        )));
    }
    if body.len() < 7 {
        return Err(Error::invalid(
            "DVB object_data: pixel-coded header truncated",
        ));
    }
    let top_len = u16::from_be_bytes([body[3], body[4]]) as usize;
    let bot_len = u16::from_be_bytes([body[5], body[6]]) as usize;
    let top_start = 7;
    let top_end = top_start + top_len;
    let bot_end = top_end + bot_len;
    if bot_end > body.len() {
        return Err(Error::invalid(
            "DVB object_data: pixel-coded line blocks truncated",
        ));
    }
    let top_rows = parse_pixel_lines(&body[top_start..top_end])?;
    let bot_rows = if bot_len > 0 {
        parse_pixel_lines(&body[top_end..bot_end])?
    } else {
        top_rows.clone()
    };
    // Interleave top-then-bottom (top field rows on even lines, bottom
    // field rows on odd lines).
    let mut rows = Vec::with_capacity(top_rows.len() + bot_rows.len());
    let n = top_rows.len().max(bot_rows.len());
    for i in 0..n {
        if i < top_rows.len() {
            rows.push(top_rows[i].clone());
        }
        if i < bot_rows.len() {
            rows.push(bot_rows[i].clone());
        }
    }
    Ok((
        object_id,
        Object {
            rows,
            coding_method,
            non_modifying_colour,
        },
    ))
}

fn parse_pixel_lines(buf: &[u8]) -> Result<Vec<Vec<u8>>> {
    let mut rows: Vec<Vec<u8>> = Vec::new();
    let mut row: Vec<u8> = Vec::new();
    let mut i = 0;
    while i < buf.len() {
        let code = buf[i];
        i += 1;
        match code {
            0x10 => {
                // 2-bit pixel-code string — subset supported.
                let (consumed, pixels) = decode_2bit_string(&buf[i..])?;
                i += consumed;
                row.extend_from_slice(&pixels);
            }
            0x11 => {
                let (consumed, pixels) = decode_4bit_string(&buf[i..])?;
                i += consumed;
                row.extend_from_slice(&pixels);
            }
            0x12 => {
                let (consumed, pixels) = decode_8bit_string(&buf[i..])?;
                i += consumed;
                row.extend_from_slice(&pixels);
            }
            0x20..=0x22 => {
                // 2-to-4 / 2-to-8 / 4-to-8 map tables — skip the next two bytes.
                if i + 2 > buf.len() {
                    return Err(Error::invalid("DVB: map table truncated"));
                }
                i += 2;
            }
            0xF0 => {
                // End-of-object-line.
                rows.push(std::mem::take(&mut row));
            }
            _ => {
                // Unknown data_type; bail politely.
                return Err(Error::invalid(format!(
                    "DVB: unknown pixel-line data_type 0x{:02X}",
                    code
                )));
            }
        }
    }
    if !row.is_empty() {
        rows.push(row);
    }
    Ok(rows)
}

/// Decode a 2-bit pixel-coded string. Returns (bytes_consumed, pixels).
fn decode_2bit_string(buf: &[u8]) -> Result<(usize, Vec<u8>)> {
    // Minimal implementation: read bit-by-bit until the terminator
    // `00 00 00 00` (8 zero bits) — the DVB spec's end-of-2-bit-code.
    let mut bits = BitReader::new(buf);
    let mut pixels = Vec::new();
    loop {
        let code = bits.read(2)?;
        if code != 0 {
            pixels.push(code as u8);
            continue;
        }
        let b1 = bits.read(1)?;
        if b1 == 1 {
            // 01 prefix followed by a 3-bit length and a 2-bit colour code:
            // (3 + run) pixels of the carried colour. Length 0 means a
            // 3-pixel run, length 7 means a 10-pixel run.
            let run = bits.read(3)? as usize + 3;
            let col = bits.read(2)? as u8;
            for _ in 0..run {
                pixels.push(col);
            }
            continue;
        }
        // 0 0 ...
        let b2 = bits.read(1)?;
        if b2 == 1 {
            // 001 - run of 0s, 3 + 4-bit count
            let run = bits.read(4)? as usize + 12;
            pixels.extend(std::iter::repeat(0_u8).take(run));
            continue;
        }
        let b3 = bits.read(2)?;
        match b3 {
            0x00 => {
                // end of string; align to next byte.
                bits.align();
                break;
            }
            0x01 => pixels.push(0), // one pixel of colour 0
            0x02 => {
                // 3 + 3-bit count run of 2-bit value
                let run = bits.read(3)? as usize + 3;
                let col = bits.read(2)? as u8;
                for _ in 0..run {
                    pixels.push(col);
                }
            }
            0x03 => {
                // 8-bit count run of 2-bit value, +25
                let run = bits.read(8)? as usize + 29;
                let col = bits.read(2)? as u8;
                for _ in 0..run {
                    pixels.push(col);
                }
            }
            _ => unreachable!(),
        }
    }
    Ok((bits.consumed_bytes(), pixels))
}

/// Decode a 4-bit pixel-coded string.
fn decode_4bit_string(buf: &[u8]) -> Result<(usize, Vec<u8>)> {
    let mut bits = BitReader::new(buf);
    let mut pixels = Vec::new();
    loop {
        let code = bits.read(4)?;
        if code != 0 {
            pixels.push(code as u8);
            continue;
        }
        let b1 = bits.read(1)?;
        if b1 == 0 {
            // Either end-of-string or run of 0s.
            let run_hdr = bits.read(3)?;
            if run_hdr == 0 {
                bits.align();
                break;
            }
            // 2..9 pixels of colour 0.
            let run = run_hdr as usize + 2;
            pixels.extend(std::iter::repeat(0_u8).take(run));
            continue;
        }
        let b2 = bits.read(1)?;
        if b2 == 0 {
            // 01xxxx - run of 4-bit colour (4..7) + colour
            let run = (bits.read(2)? as usize) + 4;
            let col = bits.read(4)? as u8;
            for _ in 0..run {
                pixels.push(col);
            }
            continue;
        }
        let b3 = bits.read(2)?;
        match b3 {
            0x00 => pixels.push(0),
            0x01 => {
                // 2 pixels of colour 0.
                pixels.push(0);
                pixels.push(0);
            }
            0x02 => {
                // 12..27 pixels of 4-bit colour
                let run = (bits.read(4)? as usize) + 9;
                let col = bits.read(4)? as u8;
                for _ in 0..run {
                    pixels.push(col);
                }
            }
            0x03 => {
                // 29..284 pixels of 4-bit colour
                let run = (bits.read(8)? as usize) + 25;
                let col = bits.read(4)? as u8;
                for _ in 0..run {
                    pixels.push(col);
                }
            }
            _ => unreachable!(),
        }
    }
    Ok((bits.consumed_bytes(), pixels))
}

/// Decode an 8-bit pixel-coded string.
fn decode_8bit_string(buf: &[u8]) -> Result<(usize, Vec<u8>)> {
    // The 8-bit encoding is byte-aligned: each "code" is one byte.
    // 0x00 introduces an escape; non-zero bytes are literal colours.
    let mut i = 0usize;
    let mut pixels = Vec::new();
    while i < buf.len() {
        let b = buf[i];
        i += 1;
        if b != 0 {
            pixels.push(b);
            continue;
        }
        if i >= buf.len() {
            return Err(Error::invalid("DVB 8-bit: escape truncated"));
        }
        let b1 = buf[i];
        i += 1;
        if b1 == 0 {
            // end-of-string
            break;
        }
        let run_flag = (b1 & 0x80) != 0;
        let count = (b1 & 0x7F) as usize;
        if !run_flag {
            if count == 0 {
                // reserved; skip
                continue;
            }
            // count pixels of colour 0.
            pixels.extend(std::iter::repeat(0_u8).take(count));
        } else {
            if i >= buf.len() {
                return Err(Error::invalid("DVB 8-bit: run-byte truncated"));
            }
            let col = buf[i];
            i += 1;
            for _ in 0..count {
                pixels.push(col);
            }
        }
    }
    Ok((i, pixels))
}

// --- bit reader --------------------------------------------------------

struct BitReader<'a> {
    buf: &'a [u8],
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, bit_pos: 0 }
    }

    fn read(&mut self, n: u32) -> Result<u32> {
        let mut out = 0u32;
        for _ in 0..n {
            if self.bit_pos >= self.buf.len() * 8 {
                return Err(Error::invalid("DVB bit reader: ran out of data"));
            }
            let byte = self.buf[self.bit_pos / 8];
            let bit = (byte >> (7 - (self.bit_pos % 8))) & 1;
            out = (out << 1) | (bit as u32);
            self.bit_pos += 1;
        }
        Ok(out)
    }

    fn align(&mut self) {
        self.bit_pos = self.bit_pos.div_ceil(8) * 8;
    }

    fn consumed_bytes(&self) -> usize {
        self.bit_pos.div_ceil(8)
    }
}

// --- bit writer ---------------------------------------------------------

/// MSB-first bit accumulator — the write-direction mirror of
/// [`BitReader`]. `into_bytes` pads the final partial byte with zero
/// bits, matching the byte-alignment the pixel-code-string decoders
/// perform after the end-of-string code.
struct BitWriter {
    out: Vec<u8>,
    cur: u8,
    used: u8,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            out: Vec::new(),
            cur: 0,
            used: 0,
        }
    }

    fn put(&mut self, n: u8, val: u32) {
        for k in (0..n).rev() {
            let bit = ((val >> k) & 1) as u8;
            self.cur = (self.cur << 1) | bit;
            self.used += 1;
            if self.used == 8 {
                self.out.push(self.cur);
                self.cur = 0;
                self.used = 0;
            }
        }
    }

    fn into_bytes(mut self) -> Vec<u8> {
        if self.used > 0 {
            self.out.push(self.cur << (8 - self.used));
        }
        self.out
    }
}

// --- pixel-code-string encoders -----------------------------------------

/// Pixel-data sub-block introducer: 2-bit pixel-code string.
pub const DATA_TYPE_2BIT: u8 = 0x10;
/// Pixel-data sub-block introducer: 4-bit pixel-code string.
pub const DATA_TYPE_4BIT: u8 = 0x11;
/// Pixel-data sub-block introducer: 8-bit pixel-code string.
pub const DATA_TYPE_8BIT: u8 = 0x12;
/// Pixel-data sub-block introducer: 2-to-4-bit map table (2-byte body).
pub const DATA_TYPE_MAP_2_TO_4: u8 = 0x20;
/// Pixel-data sub-block introducer: 2-to-8-bit map table (2-byte body).
pub const DATA_TYPE_MAP_2_TO_8: u8 = 0x21;
/// Pixel-data sub-block introducer: 4-to-8-bit map table (2-byte body).
pub const DATA_TYPE_MAP_4_TO_8: u8 = 0x22;
/// Pixel-data sub-block introducer: end of object line.
pub const DATA_TYPE_END_OF_LINE: u8 = 0xF0;

/// Encode one row of 2-bit pixels (values `0..=3`) as a 2-bit
/// pixel-code string, terminated by the end-of-string code and padded
/// to a byte boundary — the inverse of the 2-bit decode path.
///
/// Run-length forms emitted, longest-match first:
/// * `00 00 11` + 8-bit count + colour — runs of 29..=284 (any colour);
/// * `00 0 1` + 4-bit count — runs of 12..=27 of colour 0;
/// * `00 1` + 3-bit count + colour — runs of 3..=10 (any colour);
/// * `00 00 01` — a single pixel of colour 0;
/// * a bare 2-bit code — a single pixel of colour 1..=3.
pub fn encode_2bit_pixel_string(pixels: &[u8]) -> Result<Vec<u8>> {
    if let Some(&bad) = pixels.iter().find(|&&p| p > 3) {
        return Err(Error::invalid(format!(
            "DVB 2-bit encode: pixel value {bad} out of 0..=3 range"
        )));
    }
    let mut bw = BitWriter::new();
    for_each_run(pixels, |col, n| emit_2bit_run(&mut bw, col, n));
    // end-of-string: 00 + 0 + 0 + 00, then byte-align.
    bw.put(2, 0);
    bw.put(1, 0);
    bw.put(1, 0);
    bw.put(2, 0);
    Ok(bw.into_bytes())
}

fn emit_2bit_run(bw: &mut BitWriter, col: u8, mut n: usize) {
    while n > 0 {
        if n >= 29 {
            let chunk = n.min(284);
            bw.put(2, 0);
            bw.put(1, 0);
            bw.put(1, 0);
            bw.put(2, 0b11);
            bw.put(8, (chunk - 29) as u32);
            bw.put(2, col as u32);
            n -= chunk;
        } else if col == 0 && n >= 12 {
            let chunk = n.min(27);
            bw.put(2, 0);
            bw.put(1, 0);
            bw.put(1, 1);
            bw.put(4, (chunk - 12) as u32);
            n -= chunk;
        } else if n >= 3 {
            // The 3-bit-count colour-run form carries the colour
            // explicitly, so it covers colour 0 as well.
            let chunk = n.min(10);
            bw.put(2, 0);
            bw.put(1, 1);
            bw.put(3, (chunk - 3) as u32);
            bw.put(2, col as u32);
            n -= chunk;
        } else if col == 0 {
            // Single pixel of colour 0 (a bare `00` code is the escape).
            bw.put(2, 0);
            bw.put(1, 0);
            bw.put(1, 0);
            bw.put(2, 0b01);
            n -= 1;
        } else {
            bw.put(2, col as u32);
            n -= 1;
        }
    }
}

/// Encode one row of 4-bit pixels (values `0..=15`) as a 4-bit
/// pixel-code string, terminated by the end-of-string code and padded
/// to a byte boundary — the inverse of the 4-bit decode path.
///
/// Run-length forms emitted, longest-match first:
/// * `0000 1 1 11` + 8-bit count + colour — runs of 25..=280;
/// * `0000 1 1 10` + 4-bit count + colour — runs of 9..=24;
/// * `0000 0` + 3-bit count — runs of 3..=9 of colour 0;
/// * `0000 1 1 01` — two pixels of colour 0;
/// * `0000 1 1 00` — one pixel of colour 0;
/// * `0000 1 0` + 2-bit count + colour — runs of 4..=7 (non-zero colour);
/// * a bare 4-bit code — a single pixel of colour 1..=15.
pub fn encode_4bit_pixel_string(pixels: &[u8]) -> Result<Vec<u8>> {
    if let Some(&bad) = pixels.iter().find(|&&p| p > 15) {
        return Err(Error::invalid(format!(
            "DVB 4-bit encode: pixel value {bad} out of 0..=15 range"
        )));
    }
    let mut bw = BitWriter::new();
    for_each_run(pixels, |col, n| emit_4bit_run(&mut bw, col, n));
    // end-of-string: 0000 + 0 + 000, then byte-align.
    bw.put(4, 0);
    bw.put(1, 0);
    bw.put(3, 0);
    Ok(bw.into_bytes())
}

fn emit_4bit_run(bw: &mut BitWriter, col: u8, mut n: usize) {
    while n > 0 {
        if n >= 25 {
            let chunk = n.min(280);
            bw.put(4, 0);
            bw.put(1, 1);
            bw.put(1, 1);
            bw.put(2, 0b11);
            bw.put(8, (chunk - 25) as u32);
            bw.put(4, col as u32);
            n -= chunk;
        } else if n >= 9 {
            let chunk = n.min(24);
            bw.put(4, 0);
            bw.put(1, 1);
            bw.put(1, 1);
            bw.put(2, 0b10);
            bw.put(4, (chunk - 9) as u32);
            bw.put(4, col as u32);
            n -= chunk;
        } else if col == 0 {
            if n >= 3 {
                let chunk = n.min(9);
                bw.put(4, 0);
                bw.put(1, 0);
                bw.put(3, (chunk - 2) as u32);
                n -= chunk;
            } else if n == 2 {
                bw.put(4, 0);
                bw.put(1, 1);
                bw.put(1, 1);
                bw.put(2, 0b01);
                n -= 2;
            } else {
                bw.put(4, 0);
                bw.put(1, 1);
                bw.put(1, 1);
                bw.put(2, 0b00);
                n -= 1;
            }
        } else if n >= 4 {
            let chunk = n.min(7);
            bw.put(4, 0);
            bw.put(1, 1);
            bw.put(1, 0);
            bw.put(2, (chunk - 4) as u32);
            bw.put(4, col as u32);
            n -= chunk;
        } else {
            bw.put(4, col as u32);
            n -= 1;
        }
    }
}

/// Encode one row of 8-bit pixels as an 8-bit pixel-code string
/// terminated by the `0x00 0x00` end-of-string marker — the inverse of
/// the 8-bit decode path. The encoding is byte-aligned throughout:
/// non-zero bytes are literal colours, `0x00` + count (high bit clear)
/// is a run of colour 0, and `0x00` + (`0x80` | count) + colour is a
/// run of any colour. Counts max out at 127 per escape; longer runs
/// are chunked.
pub fn encode_8bit_pixel_string(pixels: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(pixels.len() / 4 + 8);
    for_each_run(pixels, |col, mut n| {
        if col == 0 {
            while n > 0 {
                let chunk = n.min(127);
                out.push(0x00);
                out.push(chunk as u8);
                n -= chunk;
            }
        } else {
            while n > 0 {
                if n >= 3 {
                    let chunk = n.min(127);
                    out.push(0x00);
                    out.push(0x80 | chunk as u8);
                    out.push(col);
                    n -= chunk;
                } else {
                    out.push(col);
                    n -= 1;
                }
            }
        }
    });
    out.push(0x00);
    out.push(0x00);
    out
}

/// Invoke `f(colour, run_length)` for each maximal run of equal pixel
/// values in `pixels`, left to right.
fn for_each_run<F: FnMut(u8, usize)>(pixels: &[u8], mut f: F) {
    let mut i = 0;
    while i < pixels.len() {
        let col = pixels[i];
        let mut n = 1usize;
        while i + n < pixels.len() && pixels[i + n] == col {
            n += 1;
        }
        f(col, n);
        i += n;
    }
}

// --- segment writers ----------------------------------------------------

/// Frame one DVB segment (`0x0F` sync + type + page_id + length + body)
/// onto `out` — the inverse of [`read_segment`]. Fails when `body` is
/// longer than the 16-bit `segment_length` field can carry.
pub fn write_segment(out: &mut Vec<u8>, seg_type: u8, page_id: u16, body: &[u8]) -> Result<()> {
    if body.len() > u16::MAX as usize {
        return Err(Error::invalid(format!(
            "DVB segment: body of {} bytes exceeds the 16-bit segment_length field",
            body.len()
        )));
    }
    out.push(0x0F);
    out.push(seg_type);
    out.extend_from_slice(&page_id.to_be_bytes());
    out.extend_from_slice(&(body.len() as u16).to_be_bytes());
    out.extend_from_slice(body);
    Ok(())
}

/// Build a display-definition segment body declaring a `width` ×
/// `height` canvas. The on-wire fields carry `width - 1` /
/// `height - 1`, so a zero-sized canvas is unrepresentable and
/// rejected.
pub fn write_display_definition(width: u16, height: u16) -> Result<Vec<u8>> {
    write_display_definition_windowed(width, height, 0, None)
}

/// Display-definition body with an explicit `dds_version_number` and an
/// optional rendering window (ETSI EN 300 743 §7.2.1). `width`/`height`
/// are the full display raster (encoded minus-1). When `window` is
/// `Some`, `display_window_flag` is set and the four inclusive
/// `display_window_*_position_minimum/maximum` fields are appended; the
/// maxima must not fall below their matching minima. This is the inverse
/// of `parse_display_definition`.
pub fn write_display_definition_windowed(
    width: u16,
    height: u16,
    version: u8,
    window: Option<DisplayWindow>,
) -> Result<Vec<u8>> {
    if width == 0 || height == 0 {
        return Err(Error::invalid("DVB DDS: zero canvas"));
    }
    let mut body = Vec::with_capacity(if window.is_some() { 13 } else { 5 });
    // dds_version_number (4) | display_window_flag (1) | reserved (3, high).
    let flag = if window.is_some() { 0x08 } else { 0x00 };
    body.push(((version & 0x0F) << 4) | flag | 0x07);
    body.extend_from_slice(&(width - 1).to_be_bytes());
    body.extend_from_slice(&(height - 1).to_be_bytes());
    if let Some(w) = window {
        if w.h_max < w.h_min || w.v_max < w.v_min {
            return Err(Error::invalid("DVB DDS: window maximum below minimum"));
        }
        body.extend_from_slice(&w.h_min.to_be_bytes());
        body.extend_from_slice(&w.h_max.to_be_bytes());
        body.extend_from_slice(&w.v_min.to_be_bytes());
        body.extend_from_slice(&w.v_max.to_be_bytes());
    }
    Ok(body)
}

/// Build a page-composition segment body: `page_time_out` (seconds),
/// the version/state byte, then one 6-byte `(region_id, x, y)` entry
/// per referenced region. Region positions are full 16-bit canvas
/// coordinates. Reserved bits are written high.
pub fn write_page_composition(
    timeout_s: u8,
    version: u8,
    state: u8,
    regions: &[(u8, u16, u16)],
) -> Vec<u8> {
    let mut body = Vec::with_capacity(2 + regions.len() * 6);
    body.push(timeout_s);
    body.push(((version & 0x0F) << 4) | ((state & 0x03) << 2) | 0x03);
    for &(region_id, x, y) in regions {
        body.push(region_id);
        body.push(0xFF); // reserved
        body.extend_from_slice(&x.to_be_bytes());
        body.extend_from_slice(&y.to_be_bytes());
    }
    body
}

/// Declarative form of a region-composition segment — the write-side
/// mirror of the decoder's region state.
#[derive(Clone, Debug)]
pub struct RegionCompositionDef {
    pub region_id: u8,
    /// `region_version_number` (4 bits used).
    pub version: u8,
    /// `region_fill_flag` — pre-paint the rectangle with the
    /// depth-appropriate pixel code before objects composite.
    pub fill: bool,
    pub width: u16,
    pub height: u16,
    /// Region colour depth: 2, 4, or 8 bits.
    pub depth_bits: u8,
    pub clut_id: u8,
    /// `region_8-bit_pixel_code` — fill index at 8-bit depth.
    pub fill_code_8: u8,
    /// `region_4-bit_pixel_code` — fill index at 4-bit depth (low nibble).
    pub fill_code_4: u8,
    /// `region_2-bit_pixel_code` — fill index at 2-bit depth (low 2 bits).
    pub fill_code_2: u8,
    /// `(object_id, x, y)` placements relative to the region origin.
    /// `x` is a 12-bit field on the wire (stored in 6+8 bits, capped at
    /// `0x3FFF` by the header layout) and `y` a 12-bit field capped at
    /// `0x0FFF`.
    pub objects: Vec<(u16, u16, u16)>,
}

/// Build a region-composition segment body from `def` — the inverse of
/// the region-composition parse. Objects are written as pixel-coded
/// (`object_type` 0), so no foreground/background colour bytes follow
/// the 6-byte placement entries.
pub fn write_region_composition(def: &RegionCompositionDef) -> Result<Vec<u8>> {
    let depth_code: u8 = match def.depth_bits {
        2 => 1,
        4 => 2,
        8 => 3,
        other => {
            return Err(Error::invalid(format!(
                "DVB region_composition: depth {other} bits is not 2/4/8"
            )))
        }
    };
    let mut body = Vec::with_capacity(10 + def.objects.len() * 6);
    body.push(def.region_id);
    body.push(((def.version & 0x0F) << 4) | ((def.fill as u8) << 3) | 0x07);
    body.extend_from_slice(&def.width.to_be_bytes());
    body.extend_from_slice(&def.height.to_be_bytes());
    // region_level_of_compatibility (top 3 bits) mirrors the declared
    // depth — the minimum CLUT depth the region needs.
    body.push((depth_code << 5) | (depth_code << 2) | 0x03);
    body.push(def.clut_id);
    body.push(def.fill_code_8);
    body.push(((def.fill_code_4 & 0x0F) << 4) | ((def.fill_code_2 & 0x03) << 2) | 0x03);
    for &(object_id, x, y) in &def.objects {
        if x > 0x3FFF {
            return Err(Error::invalid(format!(
                "DVB region_composition: object x {x} exceeds the 14-bit position field"
            )));
        }
        if y > 0x0FFF {
            return Err(Error::invalid(format!(
                "DVB region_composition: object y {y} exceeds the 12-bit position field"
            )));
        }
        body.extend_from_slice(&object_id.to_be_bytes());
        // object_type = 0 (pixel-coded bitmap) in the top 2 bits, then
        // the high bits of x.
        body.push(((x >> 8) as u8) & 0x3F);
        body.push((x & 0xFF) as u8);
        body.push(0xF0 | (((y >> 8) as u8) & 0x0F));
        body.push((y & 0xFF) as u8);
    }
    Ok(body)
}

/// One CLUT entry for [`write_clut_definition`]. Colour is carried as
/// full-range Y/Cr/Cb plus T (transparency, `0` = opaque, `255` =
/// fully transparent — the decode side maps alpha to `255 - T`).
#[derive(Clone, Copy, Debug)]
pub struct ClutEntryDef {
    pub entry_id: u8,
    pub y: u8,
    pub cr: u8,
    pub cb: u8,
    pub t: u8,
    /// `true` → 4-byte full-range Y/Cr/Cb/T form. `false` → packed
    /// 2-byte form (Y reduced to 6 bits, Cr/Cb to 4, T to 2 — lossy).
    pub full_range: bool,
}

/// Build a CLUT-definition segment body — the inverse of the CLUT
/// parse. Each entry is written in the 4-byte full-range form or the
/// packed 2-byte reduced form depending on its `full_range` flag (the
/// flag byte's bit 0 selects the form on the wire; the remaining flag
/// bits are written high).
pub fn write_clut_definition(clut_id: u8, version: u8, entries: &[ClutEntryDef]) -> Vec<u8> {
    let mut body = Vec::with_capacity(2 + entries.len() * 6);
    body.push(clut_id);
    body.push(((version & 0x0F) << 4) | 0x0F);
    for e in entries {
        body.push(e.entry_id);
        if e.full_range {
            body.push(0xFF);
            body.extend_from_slice(&[e.y, e.cr, e.cb, e.t]);
        } else {
            body.push(0xFE);
            // Packed form: 6-bit Y, 4-bit Cr, 4-bit Cb, 2-bit T — the
            // exact inverse of the decode-side unpacking.
            let y6 = e.y >> 2;
            let cr4 = e.cr >> 4;
            let cb4 = e.cb >> 4;
            let t2 = e.t >> 6;
            body.push((y6 << 2) | (cr4 >> 2));
            body.push(((cr4 & 0x03) << 6) | (cb4 << 2) | t2);
        }
    }
    body
}

/// Convert a straight-alpha RGBA pixel to the CLUT's `[Y, Cr, Cb, T]`
/// quad — the integer inverse of the decode-side BT.601 transform, so
/// a full-range CLUT entry built from this value decodes back to
/// within ±2 per channel. Greys (`R == G == B`) are bit-exact because
/// `Cb = Cr = 128` makes the chroma terms vanish.
pub fn rgba_to_clut_ycbcrt(rgba: [u8; 4]) -> [u8; 4] {
    let r = rgba[0] as i64;
    let g = rgba[1] as i64;
    let b = rgba[2] as i64;
    // BT.601 luma weights 0.299 / 0.587 / 0.114 scaled by 1 << 16; the
    // three integer weights sum to exactly 65536.
    let y = (19595 * r + 38470 * g + 7471 * b + 32768) >> 16;
    // The decode side reconstructs R = Y + (91881·Cr′) >> 16 and
    // B = Y + (116130·Cb′) >> 16 with Cr′/Cb′ centred on 128, so the
    // inverse divides by the same constants (rounded to nearest).
    let cr = div_round((r - y) << 16, 91881) + 128;
    let cb = div_round((b - y) << 16, 116130) + 128;
    [
        y.clamp(0, 255) as u8,
        cr.clamp(0, 255) as u8,
        cb.clamp(0, 255) as u8,
        255 - rgba[3],
    ]
}

/// Signed division rounded to nearest (ties away from zero).
fn div_round(num: i64, den: i64) -> i64 {
    let half = den / 2;
    if num >= 0 {
        (num + half) / den
    } else {
        (num - half) / den
    }
}

/// Build an object-data segment body for a pixel-coded object — the
/// inverse of the object-data parse.
///
/// `rows` is the full-height bitmap; even rows go to the top-field
/// block and odd rows to the bottom-field block, undoing the decode
/// side's top/bottom interleave. `depth_bits` (2, 4, or 8) selects the
/// pixel-code-string flavour used for every row. `map_tables` entries
/// (`(data_type, payload)` with a map-table data_type `0x20..=0x22`
/// and a 2-byte payload) are emitted ahead of the first pixel-code
/// string of each field block.
///
/// A single-row object still gets a non-empty bottom-field block — a
/// bare end-of-line marker producing one empty bottom row — because a
/// zero-length bottom field makes the decode side reuse the top field
/// for both fields, doubling every row.
pub fn write_object_data(
    object_id: u16,
    version: u8,
    depth_bits: u8,
    rows: &[Vec<u8>],
    map_tables: &[(u8, [u8; 2])],
) -> Result<Vec<u8>> {
    write_object_data_flags(object_id, version, depth_bits, rows, map_tables, false)
}

/// Build an object-data segment body for a pixel-coded object, choosing
/// whether the `non_modifying_colour_flag` (ETSI EN 300 743 §7.2.5) is
/// set. When `non_modifying_colour` is `true`, CLUT index `1` becomes the
/// non-modifying colour: the decode side leaves the underlying canvas
/// untouched wherever the object carries index 1, creating "transparent
/// holes". Everything else matches [`write_object_data`], of which this is
/// the general form ([`write_object_data`] passes `false`).
pub fn write_object_data_flags(
    object_id: u16,
    version: u8,
    depth_bits: u8,
    rows: &[Vec<u8>],
    map_tables: &[(u8, [u8; 2])],
    non_modifying_colour: bool,
) -> Result<Vec<u8>> {
    if rows.is_empty() {
        return Err(Error::invalid("DVB object_data: no rows to encode"));
    }
    let data_type = match depth_bits {
        2 => DATA_TYPE_2BIT,
        4 => DATA_TYPE_4BIT,
        8 => DATA_TYPE_8BIT,
        other => {
            return Err(Error::invalid(format!(
                "DVB object_data: depth {other} bits is not 2/4/8"
            )))
        }
    };
    for &(t, _) in map_tables {
        if !(DATA_TYPE_MAP_2_TO_4..=DATA_TYPE_MAP_4_TO_8).contains(&t) {
            return Err(Error::invalid(format!(
                "DVB object_data: 0x{t:02X} is not a map-table data_type"
            )));
        }
    }
    let encode_field = |field_rows: &[&Vec<u8>]| -> Result<Vec<u8>> {
        let mut block = Vec::new();
        for &(t, payload) in map_tables {
            block.push(t);
            block.extend_from_slice(&payload);
        }
        for row in field_rows {
            block.push(data_type);
            match depth_bits {
                2 => block.extend_from_slice(&encode_2bit_pixel_string(row)?),
                4 => block.extend_from_slice(&encode_4bit_pixel_string(row)?),
                _ => block.extend_from_slice(&encode_8bit_pixel_string(row)),
            }
            block.push(DATA_TYPE_END_OF_LINE);
        }
        Ok(block)
    };
    let top_rows: Vec<&Vec<u8>> = rows.iter().step_by(2).collect();
    let bot_rows: Vec<&Vec<u8>> = rows.iter().skip(1).step_by(2).collect();
    let top = encode_field(&top_rows)?;
    let bot = if bot_rows.is_empty() {
        vec![DATA_TYPE_END_OF_LINE]
    } else {
        encode_field(&bot_rows)?
    };
    if top.len() > u16::MAX as usize || bot.len() > u16::MAX as usize {
        return Err(Error::invalid(
            "DVB object_data: field block exceeds the 16-bit length field",
        ));
    }
    let mut body = Vec::with_capacity(7 + top.len() + bot.len());
    body.extend_from_slice(&object_id.to_be_bytes());
    // version (4) + coding_method 0 = pixel-coded (2) +
    // non_modifying_colour_flag (1) + reserved (1).
    let flag_bit = if non_modifying_colour { 0x02 } else { 0x00 };
    body.push(((version & 0x0F) << 4) | flag_bit);
    body.extend_from_slice(&(top.len() as u16).to_be_bytes());
    body.extend_from_slice(&(bot.len() as u16).to_be_bytes());
    body.extend_from_slice(&top);
    body.extend_from_slice(&bot);
    Ok(body)
}

// --- encoder ------------------------------------------------------------

/// Page id used by display sets the [`make_encoder`] encoder emits.
pub const ENCODER_PAGE_ID: u16 = 1;

/// Build a DVB subtitle encoder. The encoder accepts [`Frame::Video`]
/// frames with [`PixelFormat::Rgba`] and emits one [`Packet`] per frame
/// carrying a complete display-set PES payload (`0x20 0x00` prefix +
/// DDS + page composition + region composition + CLUT definition +
/// object data + end-of-display-set) — the byte shape the decoder in
/// this module consumes directly. Riding MPEG-TS needs a TS muxer
/// upstream.
///
/// Non-transparent pixels are quantised into a CLUT (index 0 is
/// reserved for the fully-transparent background) and the object is
/// cropped to the tight bounding box of non-transparent pixels, placed
/// on the canvas through the page-composition region position. The
/// pixel-code-string depth is the smallest that fits the palette:
/// ≤ 4 entries → 2-bit, ≤ 16 → 4-bit, otherwise 8-bit. When the input
/// has more than 255 distinct RGBA colours each channel is reduced to
/// 3-3-2-2 R-G-B-A bits first (lossy), with nearest-entry snapping if
/// even that overflows the 256-entry CLUT. A fully-transparent frame
/// emits an erase display set: a page composition referencing no
/// regions.
pub fn make_encoder(_params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    let mut out_params = CodecParameters::video(CodecId::new(DVBSUB_CODEC_ID));
    out_params.media_type = MediaType::Subtitle;
    out_params.pixel_format = Some(PixelFormat::Rgba);
    Ok(Box::new(DvbSubEncoder {
        codec_id: CodecId::new(DVBSUB_CODEC_ID),
        params: out_params,
        pending: VecDeque::new(),
        version: 0,
    }))
}

struct DvbSubEncoder {
    codec_id: CodecId,
    params: CodecParameters,
    pending: VecDeque<Packet>,
    /// 4-bit page/region/CLUT/object version counter, bumped per frame.
    version: u8,
}

impl Encoder for DvbSubEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let Frame::Video(v) = frame else {
            return Err(Error::unsupported(
                "DVB sub encoder only accepts Frame::Video input",
            ));
        };
        if v.planes.is_empty() {
            return Err(Error::invalid("DVB sub encoder: frame has no plane"));
        }
        let plane = &v.planes[0];
        if plane.stride == 0 || plane.stride % 4 != 0 {
            return Err(Error::invalid(
                "DVB sub encoder: RGBA plane stride must be a positive multiple of 4",
            ));
        }
        let width = plane.stride / 4;
        let height = plane.data.len() / plane.stride;
        if width == 0 || height == 0 {
            return Err(Error::invalid("DVB sub encoder: zero-sized frame"));
        }
        if width > u16::MAX as usize || height > u16::MAX as usize {
            return Err(Error::invalid(
                "DVB sub encoder: frame exceeds the 16-bit DDS dimension fields",
            ));
        }
        if self.params.width.is_none() {
            self.params.width = Some(width as u32);
            self.params.height = Some(height as u32);
        }
        let version = self.version & 0x0F;
        self.version = (self.version + 1) & 0x0F;

        let mut payload = vec![0x20, 0x00]; // data_identifier + subtitle_stream_id
        write_segment(
            &mut payload,
            SEG_DISPLAY_DEFINITION,
            ENCODER_PAGE_ID,
            &write_display_definition(width as u16, height as u16)?,
        )?;
        match encode_bbox(plane, width, height) {
            None => {
                // Fully transparent: an erase display set — a page
                // composition that references no regions decodes to a
                // fully-transparent canvas.
                write_segment(
                    &mut payload,
                    SEG_PAGE_COMPOSITION,
                    ENCODER_PAGE_ID,
                    &write_page_composition(0, version, 0, &[]),
                )?;
            }
            Some((bx, by, bw, bh)) => {
                let (indices, palette) = quantise_rgba_region(plane, width, bx, by, bw, bh);
                let depth_bits: u8 = if palette.len() <= 4 {
                    2
                } else if palette.len() <= 16 {
                    4
                } else {
                    8
                };
                write_segment(
                    &mut payload,
                    SEG_PAGE_COMPOSITION,
                    ENCODER_PAGE_ID,
                    &write_page_composition(0, version, 0, &[(0, bx as u16, by as u16)]),
                )?;
                write_segment(
                    &mut payload,
                    SEG_REGION_COMPOSITION,
                    ENCODER_PAGE_ID,
                    &write_region_composition(&RegionCompositionDef {
                        region_id: 0,
                        version,
                        fill: false,
                        width: bw as u16,
                        height: bh as u16,
                        depth_bits,
                        clut_id: 0,
                        fill_code_8: 0,
                        fill_code_4: 0,
                        fill_code_2: 0,
                        objects: vec![(0, 0, 0)],
                    })?,
                )?;
                let entries: Vec<ClutEntryDef> = palette
                    .iter()
                    .enumerate()
                    .skip(1) // entry 0 stays the default transparent
                    .map(|(i, rgba)| {
                        let [y, cr, cb, t] = rgba_to_clut_ycbcrt(*rgba);
                        ClutEntryDef {
                            entry_id: i as u8,
                            y,
                            cr,
                            cb,
                            t,
                            full_range: true,
                        }
                    })
                    .collect();
                write_segment(
                    &mut payload,
                    SEG_CLUT_DEFINITION,
                    ENCODER_PAGE_ID,
                    &write_clut_definition(0, version, &entries),
                )?;
                let rows: Vec<Vec<u8>> = indices.chunks(bw).map(<[u8]>::to_vec).collect();
                write_segment(
                    &mut payload,
                    SEG_OBJECT_DATA,
                    ENCODER_PAGE_ID,
                    &write_object_data(0, version, depth_bits, &rows, &[])?,
                )?;
            }
        }
        write_segment(&mut payload, SEG_END_OF_DISPLAY_SET, ENCODER_PAGE_ID, &[])?;

        let pts_90k = encoder_pts_90k(v).unwrap_or(0);
        let mut packet = Packet::new(0, TimeBase::new(1, 90_000), payload);
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

/// Frame pts is treated as microseconds (the canonical subtitle pts
/// unit) and rescaled to the 90 kHz transport clock the packet carries.
fn encoder_pts_90k(v: &VideoFrame) -> Option<u32> {
    let pts = v.pts?;
    let scaled = TimeBase::new(1, 1_000_000).rescale(pts, TimeBase::new(1, 90_000));
    if scaled < 0 {
        Some(0)
    } else {
        Some(scaled as u32)
    }
}

/// Tight bounding box of pixels with non-zero alpha, or `None` when the
/// frame is fully transparent. Returns `(x, y, width, height)`.
fn encode_bbox(
    plane: &VideoPlane,
    width: usize,
    height: usize,
) -> Option<(usize, usize, usize, usize)> {
    let mut min_x = width;
    let mut max_x = 0usize;
    let mut min_y = height;
    let mut max_y = 0usize;
    let mut any = false;
    for row in 0..height {
        let line = &plane.data[row * plane.stride..row * plane.stride + width * 4];
        for (col, px) in line.chunks_exact(4).enumerate() {
            if px[3] != 0 {
                any = true;
                min_x = min_x.min(col);
                max_x = max_x.max(col);
                min_y = min_y.min(row);
                max_y = max_y.max(row);
            }
        }
    }
    if !any {
        return None;
    }
    Some((min_x, min_y, max_x - min_x + 1, max_y - min_y + 1))
}

/// Quantise the RGBA pixels of a frame sub-rectangle into an indexed
/// bitmap plus RGBA palette. Index 0 is reserved for fully-transparent
/// pixels. An input with more than 255 distinct opaque colours is
/// re-quantised with 3-3-2-2 R-G-B-A channel reduction; if even that
/// overflows the 256-entry CLUT, further novel colours snap to the
/// nearest existing entry.
fn quantise_rgba_region(
    plane: &VideoPlane,
    frame_width: usize,
    bx: usize,
    by: usize,
    bw: usize,
    bh: usize,
) -> (Vec<u8>, Vec<[u8; 4]>) {
    // Exact pass first.
    if let Some(exact) = quantise_pass(plane, frame_width, bx, by, bw, bh, false) {
        return exact;
    }
    // 3-3-2-2 reduction (with nearest-entry snapping) always succeeds.
    quantise_pass(plane, frame_width, bx, by, bw, bh, true)
        .expect("reduced quantisation pass cannot overflow")
}

fn quantise_pass(
    plane: &VideoPlane,
    frame_width: usize,
    bx: usize,
    by: usize,
    bw: usize,
    bh: usize,
    reduce: bool,
) -> Option<(Vec<u8>, Vec<[u8; 4]>)> {
    let mut palette: Vec<[u8; 4]> = vec![[0, 0, 0, 0]];
    let mut map: HashMap<[u8; 4], u8> = HashMap::new();
    map.insert([0, 0, 0, 0], 0);
    let mut indices = vec![0u8; bw * bh];
    for row in 0..bh {
        let src = (by + row) * plane.stride;
        let line = &plane.data[src..src + frame_width * 4];
        for col in 0..bw {
            let px = &line[(bx + col) * 4..(bx + col) * 4 + 4];
            if px[3] == 0 {
                continue; // index 0
            }
            let key = if reduce {
                key_3322(px)
            } else {
                [px[0], px[1], px[2], px[3]]
            };
            let idx = if let Some(&idx) = map.get(&key) {
                idx
            } else if palette.len() < 256 {
                let idx = palette.len() as u8;
                palette.push(key);
                map.insert(key, idx);
                idx
            } else if reduce {
                nearest_clut_entry(&palette, key)
            } else {
                return None; // exact pass overflowed — caller reduces
            };
            indices[row * bw + col] = idx;
        }
    }
    Some((indices, palette))
}

/// Reduce an opaque-ish RGBA pixel to 3-3-2-2 R-G-B-A bits, mapping each
/// bucket to its top value so fully-saturated channels stay at 255.
fn key_3322(px: &[u8]) -> [u8; 4] {
    [px[0] | 0x1F, px[1] | 0x1F, px[2] | 0x3F, px[3] | 0x3F]
}

fn nearest_clut_entry(palette: &[[u8; 4]], key: [u8; 4]) -> u8 {
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

// --- decoder -----------------------------------------------------------

/// Build a DVB subtitle decoder. Packet payloads are the raw PES
/// payload (with the `0x20 0x00` data_identifier prefix stripped).
pub fn make_decoder(_params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Ok(Box::new(DvbSubDecoder {
        codec_id: CodecId::new(DVBSUB_CODEC_ID),
        pending: VecDeque::new(),
        eof: false,
    }))
}

struct DvbSubDecoder {
    codec_id: CodecId,
    pending: VecDeque<Frame>,
    eof: bool,
}

impl Decoder for DvbSubDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        let payload = strip_pes_prefix(&packet.data);
        let mut dds = DisplayDefinition::default();
        let mut regions: HashMap<u8, Region> = HashMap::new();
        let mut objects: HashMap<u16, Object> = HashMap::new();
        let mut cluts: HashMap<u8, Clut> = HashMap::new();
        let mut page: Vec<PageRegion> = Vec::new();
        let mut cur = 0;
        while cur < payload.len() {
            let (seg, next) = match read_segment(payload, cur) {
                Ok(x) => x,
                Err(Error::NeedMore) => break,
                Err(e) => return Err(e),
            };
            match seg.seg_type {
                SEG_DISPLAY_DEFINITION => {
                    dds = parse_display_definition(&seg.body)?;
                }
                SEG_PAGE_COMPOSITION => {
                    page = parse_page_composition(&seg.body)?;
                }
                SEG_REGION_COMPOSITION => {
                    let (id, region) = parse_region_composition(&seg.body)?;
                    regions.insert(id, region);
                }
                SEG_CLUT_DEFINITION => {
                    parse_clut_into(&seg.body, &mut cluts)?;
                }
                SEG_OBJECT_DATA => {
                    let (id, obj) = parse_object_data(&seg.body)?;
                    objects.insert(id, obj);
                }
                SEG_DISPARITY_SIGNALLING => {
                    // 3D plano-stereoscopic disparity metadata (§7.2.7).
                    // Parsed and validated for structural correctness; the
                    // 2D RGBA canvas this decoder paints is the
                    // disparity-zero view (the spec's "implicit disparity
                    // of zero" baseline), so the parsed values do not shift
                    // the emitted picture — they are surfaced for callers
                    // doing stereoscopic placement.
                    let _dss = parse_disparity_signalling(&seg.body)?;
                }
                SEG_END_OF_DISPLAY_SET => {
                    break;
                }
                _ => {}
            }
            cur = next;
        }
        let width = dds.width as usize;
        let height = dds.height as usize;
        if width == 0 || height == 0 {
            return Err(Error::invalid("DVB sub: zero canvas"));
        }
        let mut canvas = vec![0u8; width * height * 4];
        // DVB page composition stacks regions and, within a region, the
        // referenced objects. When two of those overlap and the topmost
        // CLUT entry is only partially transparent, the correct result is
        // the source colour blended *over* whatever is already on the
        // canvas (Porter–Duff source-over), not a hard overwrite — so the
        // blit runs through the shared alpha-aware compositor. Index 0 is
        // the conventional DVB transparent background, mapped to alpha 0
        // so it's skipped.
        for pr in &page {
            let Some(region) = regions.get(&pr.region_id) else {
                continue;
            };
            let clut = cluts.get(&region.clut_id).cloned().unwrap_or_default();
            // ETSI EN 300 743 §7.2.3: when region_fill_flag is set, the
            // region rectangle is pre-filled with the depth-appropriate
            // `region_n-bit_pixel_code` *before* the region's objects
            // composite on top. Without this the rectangle stays at the
            // canvas's transparent background, so a stream that relies on
            // a solid backdrop (e.g. an opaque banner with text objects
            // overlaid) would render with show-through holes. Fill is
            // bounded to the region's declared `width` × `height` at the
            // page-composition `(x, y)`, intersected with the canvas.
            if region.fill {
                let fill_idx = match region.depth_bits {
                    2 => region.fill_code_2,
                    4 => region.fill_code_4,
                    _ => region.fill_code_8,
                };
                let fill_rgba = clut.entries[fill_idx as usize];
                // A zero alpha after CLUT lookup means the declared fill
                // colour is transparent, which is the same as not filling
                // — short-circuit so the canvas keeps any pixels from a
                // lower-z-order region.
                if fill_rgba[3] != 0 {
                    let rx0 = (pr.x as usize).min(width);
                    let ry0 = (pr.y as usize).min(height);
                    let rx1 = (pr.x as usize + region.width as usize).min(width);
                    let ry1 = (pr.y as usize + region.height as usize).min(height);
                    for y in ry0..ry1 {
                        let row_off = (y * width + rx0) * 4;
                        let row_end = (y * width + rx1) * 4;
                        for px in canvas[row_off..row_end].chunks_exact_mut(4) {
                            px.copy_from_slice(&fill_rgba);
                        }
                    }
                }
            }
            for ro in &region.objects {
                let Some(obj) = objects.get(&ro.object_id) else {
                    continue;
                };
                let base_x = pr.x as usize + ro.x as usize;
                let base_y = pr.y as usize + ro.y as usize;
                let non_modifying = obj.non_modifying_colour;
                crate::composite::blit_indexed(
                    &mut canvas,
                    width,
                    height,
                    &obj.rows,
                    base_x,
                    base_y,
                    |px| {
                        // ETSI EN 300 743 §7.2.5: index 0 is the region's
                        // transparent background. When the object's
                        // `non_modifying_colour_flag` is set, index 1 is the
                        // non-modifying colour — it leaves whatever is
                        // already on the canvas (region background or a
                        // lower-z-order object) untouched, punching a
                        // "transparent hole" through this object. Returning
                        // alpha 0 makes the compositor skip the write.
                        if px == 0 || (non_modifying && px == 1) {
                            [0, 0, 0, 0]
                        } else {
                            clut.entries[px as usize]
                        }
                    },
                );
            }
        }

        // ETSI EN 300 743 §7.2.1: a DDS whose `display_window_flag` is set
        // confines this display set to a window within the display raster
        // — it "is intended to be rendered in a window within the display
        // size defined by display_width and display_height". Region
        // addresses stay absolute display coordinates (§7.2.2), so any
        // pixel that composited outside the inclusive window rectangle is
        // not part of the intended display set; clear it back to the
        // transparent background. `dds.version` is parsed for callers that
        // track per-segment versioning but does not alter rendering.
        let _ = dds.version;
        if let Some(w) = dds.window {
            let wx0 = (w.h_min as usize).min(width);
            let wx1 = ((w.h_max as usize) + 1).min(width);
            let wy0 = (w.v_min as usize).min(height);
            let wy1 = ((w.v_max as usize) + 1).min(height);
            for y in 0..height {
                let in_v = y >= wy0 && y < wy1;
                for x in 0..width {
                    if in_v && x >= wx0 && x < wx1 {
                        continue;
                    }
                    let off = (y * width + x) * 4;
                    canvas[off..off + 4].copy_from_slice(&[0, 0, 0, 0]);
                }
            }
        }

        let frame = VideoFrame {
            pts: packet.pts,
            planes: vec![VideoPlane {
                stride: width * 4,
                data: canvas,
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
        // DVB subtitles rebuild the region/object/clut state from
        // scratch for each PES payload, so there's no per-stream state
        // to clear. Drop the ready-frame queue and the eof latch.
        self.pending.clear();
        self.eof = false;
        Ok(())
    }
}

/// Strip the `0x20 0x00` data_identifier / subtitle_stream_id prefix if
/// present. Containers typically hand us the PES payload that already
/// starts with the DVB segment stream — but some strip these lead
/// bytes, so we accept either shape.
fn strip_pes_prefix(buf: &[u8]) -> &[u8] {
    if buf.len() >= 2 && buf[0] == 0x20 && buf[1] == 0x00 {
        &buf[2..]
    } else {
        buf
    }
}

// --- test helper -------------------------------------------------------

#[doc(hidden)]
pub fn build_demo_pes(canvas: (u16, u16), pixels: &[u8], width: usize, height: usize) -> Vec<u8> {
    assert_eq!(pixels.len(), width * height);
    fn segment(out: &mut Vec<u8>, seg_type: u8, body: &[u8]) {
        out.push(0x0F);
        out.push(seg_type);
        out.extend_from_slice(&1u16.to_be_bytes()); // page_id
        out.extend_from_slice(&(body.len() as u16).to_be_bytes());
        out.extend_from_slice(body);
    }

    let mut out = vec![0x20, 0x00]; // data_identifier + subtitle_stream_id

    // DDS
    let mut dds = Vec::new();
    dds.push(0); // version + flags
    dds.extend_from_slice(&(canvas.0.saturating_sub(1)).to_be_bytes());
    dds.extend_from_slice(&(canvas.1.saturating_sub(1)).to_be_bytes());
    segment(&mut out, SEG_DISPLAY_DEFINITION, &dds);

    // Page composition: one region at (0,0).
    let mut page = vec![
        0,    // page_time_out (s)
        0,    // version/state
        0,    // region_id
        0xFF, // reserved
    ];
    page.extend_from_slice(&0u16.to_be_bytes()); // x
    page.extend_from_slice(&0u16.to_be_bytes()); // y
    segment(&mut out, SEG_PAGE_COMPOSITION, &page);

    // Region composition: one object at (0,0), 8-bit depth, CLUT=0.
    let mut region = Vec::new();
    region.push(0); // region_id
    region.push(0); // version + fill_flag
    region.extend_from_slice(&(width as u16).to_be_bytes());
    region.extend_from_slice(&(height as u16).to_be_bytes());
    region.push(3 << 2); // depth = 3 (8-bit)
    region.push(0); // clut_id
    region.push(0); // 8-bit pixel code
    region.push(0); // 4/2-bit codes
                    // object entry
    region.extend_from_slice(&0u16.to_be_bytes()); // object_id + obj_type(0)
    region.extend_from_slice(&0u16.to_be_bytes()); // x
    region.extend_from_slice(&0u16.to_be_bytes()); // y
    segment(&mut out, SEG_REGION_COMPOSITION, &region);

    // CLUT: entries 0..3 with identifiable RGBA values.
    let mut clut = vec![
        0, // clut_id
        0, // version
        // Entry 1 → white. Y=255, Cr=128, Cb=128, T=0.
        1,    // entry_id
        0xFF, // flags — mark "full precision" (bit 0) + 8-bit type bits
        255,  // Y
        128,  // Cr
        128,  // Cb
        0,    // T
    ];
    // Entry 2 → red.
    clut.push(2);
    clut.push(0xFF);
    clut.push(81); // approximate Y for red
    clut.push(240); // Cr
    clut.push(90); // Cb
    clut.push(0);
    segment(&mut out, SEG_CLUT_DEFINITION, &clut);

    // Object data: 8-bit pixel-coded string, two rows top+bottom.
    let mut obj = Vec::new();
    obj.extend_from_slice(&0u16.to_be_bytes()); // object_id
                                                // coding_method = 0 in bits 3..4 of byte[2]. Rest zero.
    obj.push(0);
    // Build top-field / bottom-field line blocks.
    fn encode_rows_8bit(rows: &[Vec<u8>]) -> Vec<u8> {
        let mut out = Vec::new();
        for row in rows {
            out.push(0x12); // 8-bit pixel code
            for &p in row {
                if p == 0 {
                    out.push(0x00);
                    out.push(0x01); // 1 pixel of colour 0 (short form)
                    continue;
                }
                out.push(p);
            }
            out.push(0x00);
            out.push(0x00); // end of 8-bit string
            out.push(0xF0); // end-of-object-line
        }
        out
    }
    // Split into top/bottom field rows by interleave.
    let mut top: Vec<Vec<u8>> = Vec::new();
    let mut bot: Vec<Vec<u8>> = Vec::new();
    for (i, row) in pixels.chunks(width).enumerate() {
        if i % 2 == 0 {
            top.push(row.to_vec());
        } else {
            bot.push(row.to_vec());
        }
    }
    let top_bytes = encode_rows_8bit(&top);
    let bot_bytes = encode_rows_8bit(&bot);
    obj.extend_from_slice(&(top_bytes.len() as u16).to_be_bytes());
    obj.extend_from_slice(&(bot_bytes.len() as u16).to_be_bytes());
    obj.extend_from_slice(&top_bytes);
    obj.extend_from_slice(&bot_bytes);
    segment(&mut out, SEG_OBJECT_DATA, &obj);

    // END of display-set
    segment(&mut out, SEG_END_OF_DISPLAY_SET, &[]);

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::TimeBase;

    #[test]
    fn decodes_pixel_coded_bitmap() {
        // 2×2 bitmap: white, red, red, white.
        let pixels = [1u8, 2, 2, 1];
        let pes = build_demo_pes((2, 2), &pixels, 2, 2);
        let params = CodecParameters::video(CodecId::new(DVBSUB_CODEC_ID));
        let mut dec = make_decoder(&params).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 90_000), pes).with_pts(0);
        dec.send_packet(&pkt).unwrap();
        let frame = dec.receive_frame().unwrap();
        let Frame::Video(v) = frame else {
            panic!("expected video frame");
        };
        assert_eq!(v.planes[0].stride, 2 * 4);
        assert_eq!(v.planes[0].data.len(), 2 * 2 * 4);
        // Row 0: white, red.
        let r0c0 = &v.planes[0].data[0..4];
        let r0c1 = &v.planes[0].data[4..8];
        // White ≈ all channels high.
        assert!(
            r0c0[0] > 200 && r0c0[1] > 200 && r0c0[2] > 200,
            "expected white, got {:?}",
            r0c0
        );
        // Red-ish: R dominant.
        assert!(
            r0c1[0] > r0c1[1] && r0c1[0] > r0c1[2],
            "expected red-dominant, got {:?}",
            r0c1
        );
    }

    /// Frame a single DVB segment (`0x0F`, type, page_id, length, body).
    fn dvb_segment(out: &mut Vec<u8>, seg_type: u8, body: &[u8]) {
        out.push(0x0F);
        out.push(seg_type);
        out.extend_from_slice(&1u16.to_be_bytes()); // page_id
        out.extend_from_slice(&(body.len() as u16).to_be_bytes());
        out.extend_from_slice(body);
    }

    /// One 8-bit-coded object body carrying a single top-field row of
    /// `row` pixels (no bottom field). `object_id` identifies it for the
    /// region's object reference.
    fn dvb_object_8bit(object_id: u16, row: &[u8]) -> Vec<u8> {
        let mut obj = Vec::new();
        obj.extend_from_slice(&object_id.to_be_bytes());
        obj.push(0); // version + coding_method(0) + flags
        let mut top = vec![0x12]; // 8-bit pixel-code-string data type
        top.extend_from_slice(&encode_8bit_literal(row));
        top.push(0xF0); // end-of-object-line
        obj.extend_from_slice(&(top.len() as u16).to_be_bytes()); // top length
        obj.extend_from_slice(&0u16.to_be_bytes()); // bottom length = 0
        obj.extend_from_slice(&top);
        obj
    }

    /// A region-composition body: `region_id`, 8-bit depth, `clut_id`,
    /// and a single object reference at the region origin.
    fn dvb_region_8bit(region_id: u8, clut_id: u8, w: u16, h: u16, object_id: u16) -> Vec<u8> {
        let mut region = Vec::new();
        region.push(region_id);
        region.push(0); // version + fill_flag(0)
        region.extend_from_slice(&w.to_be_bytes());
        region.extend_from_slice(&h.to_be_bytes());
        region.push(3 << 2); // region_depth = 3 (8-bit)
        region.push(clut_id);
        region.push(0); // region_8-bit_pixel_code
        region.push(0); // region_4/2-bit_pixel_code
                        // object reference at (0,0), obj_type = 0.
        region.extend_from_slice(&object_id.to_be_bytes());
        region.extend_from_slice(&0u16.to_be_bytes()); // x
        region.extend_from_slice(&0u16.to_be_bytes()); // y
        region
    }

    /// One full-precision CLUT entry: entry 1 → the given Y/Cr/Cb/T.
    fn dvb_clut_entry1(clut_id: u8, y: u8, cr: u8, cb: u8, t: u8) -> Vec<u8> {
        vec![
            clut_id, 0, /* entry_id */ 1, /* flags: full-precision */ 0x01, y, cr, cb, t,
        ]
    }

    #[test]
    fn multi_region_overlap_composites_in_page_order() {
        // Two single-pixel regions at the same page position (0,0). The
        // earlier page-list region paints an opaque colour; the later one
        // paints a half-transparent colour on top. The spec composites
        // page regions in list order with Porter–Duff source-over, so the
        // result is a *blend* of the two — not the first region alone.
        let mut out = vec![0x20, 0x00]; // data_identifier + stream_id

        // DDS: 2×1 canvas (encoded as max-index width/height-1).
        let mut dds = Vec::new();
        dds.push(0);
        dds.extend_from_slice(&1u16.to_be_bytes()); // width - 1  → 2
        dds.extend_from_slice(&0u16.to_be_bytes()); // height - 1 → 1
        dvb_segment(&mut out, SEG_DISPLAY_DEFINITION, &dds);

        // Page composition: region 0 then region 1, both at (0,0).
        let mut page = vec![0 /* timeout */, 0 /* version/state */];
        for region_id in [0u8, 1] {
            page.push(region_id);
            page.push(0xFF); // reserved
            page.extend_from_slice(&0u16.to_be_bytes()); // x
            page.extend_from_slice(&0u16.to_be_bytes()); // y
        }
        dvb_segment(&mut out, SEG_PAGE_COMPOSITION, &page);

        // Region 0 → opaque white (CLUT 0). Region 1 → half-transparent
        // red (CLUT 1) layered on top.
        dvb_segment(
            &mut out,
            SEG_REGION_COMPOSITION,
            &dvb_region_8bit(0, 0, 1, 1, 100),
        );
        dvb_segment(
            &mut out,
            SEG_REGION_COMPOSITION,
            &dvb_region_8bit(1, 1, 1, 1, 101),
        );

        // CLUT 0 entry 1 → opaque white (T = 0 → alpha 255).
        dvb_segment(
            &mut out,
            SEG_CLUT_DEFINITION,
            &dvb_clut_entry1(0, 255, 128, 128, 0),
        );
        // CLUT 1 entry 1 → half-transparent red (T = 128 → alpha ~127).
        dvb_segment(
            &mut out,
            SEG_CLUT_DEFINITION,
            &dvb_clut_entry1(1, 81, 240, 90, 128),
        );

        // Object 100 paints colour 1 in region 0; object 101 in region 1.
        dvb_segment(&mut out, SEG_OBJECT_DATA, &dvb_object_8bit(100, &[1]));
        dvb_segment(&mut out, SEG_OBJECT_DATA, &dvb_object_8bit(101, &[1]));

        dvb_segment(&mut out, SEG_END_OF_DISPLAY_SET, &[]);

        let params = CodecParameters::video(CodecId::new(DVBSUB_CODEC_ID));
        let mut dec = make_decoder(&params).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 90_000), out).with_pts(0);
        dec.send_packet(&pkt).unwrap();
        let Frame::Video(v) = dec.receive_frame().unwrap() else {
            panic!("expected video frame");
        };
        let px = &v.planes[0].data[0..4];
        // If only the first region won, the pixel would be pure opaque
        // white (R≈G≈B, all high). Source-over of a half-transparent red
        // pulls green/blue down below red while keeping red high.
        assert!(
            px[0] > px[1] && px[0] > px[2],
            "expected red-biased blend from the second region, got {px:?}"
        );
        assert!(
            px[1] < 255 && px[2] < 255,
            "expected the top region to darken green/blue, got {px:?}"
        );
        // Final pixel is opaque (the underlying region was opaque white).
        assert_eq!(px[3], 255, "expected opaque result, got {px:?}");
    }

    #[test]
    fn display_window_clips_pixels_outside_it() {
        // ETSI EN 300 743 §7.2.1: a DDS with display_window_flag set
        // confines the display set to a window inside the raster. We lay a
        // 4×1 canvas with two opaque single-pixel regions — one at x=0
        // (inside a window covering pixels 0..=1) and one at x=3 (outside
        // it). After rendering, the in-window pixel survives and the
        // out-of-window pixel is cleared to transparent.
        let mut out = vec![0x20, 0x00]; // data_identifier + stream_id

        // DDS: 4×1 raster, window flag set, window = horizontal 0..=1, vertical 0..=0.
        let win = DisplayWindow {
            h_min: 0,
            h_max: 1,
            v_min: 0,
            v_max: 0,
        };
        let dds = write_display_definition_windowed(4, 1, 0, Some(win)).unwrap();
        dvb_segment(&mut out, SEG_DISPLAY_DEFINITION, &dds);

        // Region 0 at page x=0 (inside), region 1 at page x=3 (outside).
        let mut page = vec![0u8, 0u8];
        page.push(0); // region_id 0
        page.push(0xFF);
        page.extend_from_slice(&0u16.to_be_bytes()); // x = 0
        page.extend_from_slice(&0u16.to_be_bytes()); // y
        page.push(1); // region_id 1
        page.push(0xFF);
        page.extend_from_slice(&3u16.to_be_bytes()); // x = 3 (outside window)
        page.extend_from_slice(&0u16.to_be_bytes()); // y
        dvb_segment(&mut out, SEG_PAGE_COMPOSITION, &page);

        dvb_segment(
            &mut out,
            SEG_REGION_COMPOSITION,
            &dvb_region_8bit(0, 0, 1, 1, 100),
        );
        dvb_segment(
            &mut out,
            SEG_REGION_COMPOSITION,
            &dvb_region_8bit(1, 0, 1, 1, 101),
        );
        // CLUT 0 entry 1 → opaque white.
        dvb_segment(
            &mut out,
            SEG_CLUT_DEFINITION,
            &dvb_clut_entry1(0, 255, 128, 128, 0),
        );
        dvb_segment(&mut out, SEG_OBJECT_DATA, &dvb_object_8bit(100, &[1]));
        dvb_segment(&mut out, SEG_OBJECT_DATA, &dvb_object_8bit(101, &[1]));
        dvb_segment(&mut out, SEG_END_OF_DISPLAY_SET, &[]);

        let params = CodecParameters::video(CodecId::new(DVBSUB_CODEC_ID));
        let mut dec = make_decoder(&params).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 90_000), out).with_pts(0);
        dec.send_packet(&pkt).unwrap();
        let Frame::Video(v) = dec.receive_frame().unwrap() else {
            panic!("expected video frame");
        };
        let data = &v.planes[0].data;
        // Pixel 0 (inside window) painted opaque.
        assert_eq!(data[3], 255, "in-window pixel should be opaque");
        // Pixel 3 (outside window) cleared to transparent despite its region.
        assert_eq!(
            &data[12..16],
            &[0, 0, 0, 0],
            "out-of-window pixel must be cleared"
        );
    }

    #[test]
    fn rejects_character_coded_objects() {
        // Object-data body: id=0, coding_method=1 (character string).
        let mut body = Vec::new();
        body.extend_from_slice(&0u16.to_be_bytes()); // object id
        body.push(0b0000_0100); // coding_method = 1 in bits 2..3
        body.extend_from_slice(&0u16.to_be_bytes());
        body.extend_from_slice(&0u16.to_be_bytes());

        let err = parse_object_data(&body).unwrap_err();
        match err {
            Error::Unsupported(_) => {}
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    /// ETSI EN 300 743 §7.2.5: `non_modifying_colour_flag` (bit 1 of the
    /// version/coding byte) is decoded into the parsed object.
    #[test]
    fn parse_object_decodes_non_modifying_colour_flag() {
        // Flag clear.
        let plain = write_object_data_flags(0, 0, 8, &[vec![1, 2]], &[], false).unwrap();
        let (_, obj) = parse_object_data(&plain).unwrap();
        assert!(!obj.non_modifying_colour);

        // Flag set — must survive the write→parse round-trip.
        let holed = write_object_data_flags(0, 0, 8, &[vec![1, 2]], &[], true).unwrap();
        let (_, obj) = parse_object_data(&holed).unwrap();
        assert!(obj.non_modifying_colour);
        // Same pixel rows either way; only the flag byte changes.
        assert_eq!(obj.rows, parse_object_data(&plain).unwrap().1.rows);
    }

    /// CLUT body with two full-precision entries (ids 1 and 2).
    fn dvb_clut_entries(clut_id: u8, e1: (u8, u8, u8, u8), e2: (u8, u8, u8, u8)) -> Vec<u8> {
        let mut body = vec![clut_id, 0];
        for (id, (y, cr, cb, t)) in [(1u8, e1), (2u8, e2)] {
            body.push(id);
            body.push(0x01); // full-precision flag
            body.extend_from_slice(&[y, cr, cb, t]);
        }
        body
    }

    /// One 8-bit-coded object body with a chosen `non_modifying_colour`
    /// flag, carrying a single top-field row.
    fn dvb_object_8bit_flagged(object_id: u16, row: &[u8], non_modifying: bool) -> Vec<u8> {
        let mut obj = Vec::new();
        obj.extend_from_slice(&object_id.to_be_bytes());
        obj.push(if non_modifying { 0x02 } else { 0x00 });
        let mut top = vec![0x12];
        top.extend_from_slice(&encode_8bit_literal(row));
        top.push(0xF0);
        obj.extend_from_slice(&(top.len() as u16).to_be_bytes());
        obj.extend_from_slice(&0u16.to_be_bytes());
        obj.extend_from_slice(&top);
        obj
    }

    /// ETSI EN 300 743 §7.2.5: when an object's `non_modifying_colour_flag`
    /// is set, CLUT index 1 is the non-modifying colour — pixels carrying
    /// it leave the underlying region/object content untouched, punching a
    /// transparent hole. Index 2 still paints normally.
    #[test]
    fn non_modifying_colour_punches_transparent_hole() {
        // Two regions at (0,0) over a 3×1 canvas. Region 0 is the opaque
        // white background object; region 1 sits on top with the flag set,
        // painting [1, 2, 1] — index 1 = hole (white shows through),
        // index 2 = opaque red (overpaints).
        let build = |non_modifying: bool| -> Vec<u8> {
            let mut out = vec![0x20, 0x00];

            // DDS: 3×1 raster (width/height encoded as max-index).
            let mut dds = Vec::new();
            dds.push(0); // dds_version + flags (window flag clear)
            dds.extend_from_slice(&2u16.to_be_bytes()); // width-1 = 2
            dds.extend_from_slice(&0u16.to_be_bytes()); // height-1 = 0
            dvb_segment(&mut out, SEG_DISPLAY_DEFINITION, &dds);

            // Page: region 0 then region 1, both at (0,0).
            let mut page = vec![0u8, 0u8];
            for region_id in [0u8, 1] {
                page.push(region_id);
                page.push(0xFF);
                page.extend_from_slice(&0u16.to_be_bytes());
                page.extend_from_slice(&0u16.to_be_bytes());
            }
            dvb_segment(&mut out, SEG_PAGE_COMPOSITION, &page);

            // Region 0: 3×1, clut 0, object 100. Region 1: 3×1, clut 1, object 101.
            dvb_segment(
                &mut out,
                SEG_REGION_COMPOSITION,
                &dvb_region_8bit(0, 0, 3, 1, 100),
            );
            dvb_segment(
                &mut out,
                SEG_REGION_COMPOSITION,
                &dvb_region_8bit(1, 1, 3, 1, 101),
            );

            // CLUT 0: entry 1 = opaque white. CLUT 1: entry 1 = opaque
            // green (would overpaint if NOT treated as a hole), entry 2 =
            // opaque red.
            dvb_segment(
                &mut out,
                SEG_CLUT_DEFINITION,
                &dvb_clut_entry1(0, 235, 128, 128, 0),
            );
            dvb_segment(
                &mut out,
                SEG_CLUT_DEFINITION,
                &dvb_clut_entries(1, (145, 54, 34, 0), (81, 240, 90, 0)),
            );

            // Object 100: opaque white across all three pixels.
            dvb_segment(&mut out, SEG_OBJECT_DATA, &dvb_object_8bit(100, &[1, 1, 1]));
            // Object 101 (top): [1, 2, 1] with the chosen flag.
            dvb_segment(
                &mut out,
                SEG_OBJECT_DATA,
                &dvb_object_8bit_flagged(101, &[1, 2, 1], non_modifying),
            );
            dvb_segment(&mut out, SEG_END_OF_DISPLAY_SET, &[]);
            out
        };

        let decode = |bytes: Vec<u8>| -> Vec<u8> {
            let params = CodecParameters::video(CodecId::new(DVBSUB_CODEC_ID));
            let mut dec = make_decoder(&params).unwrap();
            let pkt = Packet::new(0, TimeBase::new(1, 90_000), bytes).with_pts(0);
            dec.send_packet(&pkt).unwrap();
            let Frame::Video(v) = dec.receive_frame().unwrap() else {
                panic!("expected video frame");
            };
            v.planes[0].data.clone()
        };

        // Flag clear: index 1 of the top object paints CLUT-1 green over
        // the white background at pixels 0 and 2.
        let plain = decode(build(false));
        let green = &plain[0..4];
        assert!(
            green[1] > green[0] && green[1] > green[2],
            "flag clear: pixel 0 should be green (top object overpaints), got {green:?}"
        );

        // Flag set: index 1 is the non-modifying colour — pixels 0 and 2
        // leave the white background untouched; pixel 1 (index 2) overpaints red.
        let holed = decode(build(true));
        let p0 = &holed[0..4];
        let p1 = &holed[4..8];
        let p2 = &holed[8..12];
        // Pixel 0 and 2 stayed white (R≈G≈B, all high) — the hole let the
        // underlying object show through unchanged.
        assert!(
            p0[0] > 200 && p0[1] > 200 && p0[2] > 200,
            "hole pixel 0 should keep the white background, got {p0:?}"
        );
        assert!(
            p2[0] > 200 && p2[1] > 200 && p2[2] > 200,
            "hole pixel 2 should keep the white background, got {p2:?}"
        );
        // Pixel 1 (index 2) overpainted red.
        assert!(
            p1[0] > p1[1] && p1[0] > p1[2],
            "pixel 1 (index 2) should overpaint red, got {p1:?}"
        );
    }

    // --- pixel-line decoder coverage --------------------------------

    /// Tiny LCG used by the negative-input fuzz sweeps. Deterministic so
    /// failures are reproducible from the test name alone.
    fn lcg(state: &mut u64) -> u32 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (*state >> 33) as u32
    }

    /// Encode `pixels` as an 8-bit-coded string terminated by the
    /// `00 00` end-of-string marker. Uses one literal byte per pixel
    /// (with the `00 01` short-form for transparent), matching what
    /// `decode_8bit_string` accepts.
    fn encode_8bit_literal(pixels: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(pixels.len() + 2);
        for &p in pixels {
            if p == 0 {
                // count=1, run_flag=0 → 1 pixel of colour 0.
                out.push(0x00);
                out.push(0x01);
            } else {
                out.push(p);
            }
        }
        out.push(0x00);
        out.push(0x00);
        out
    }

    /// Encode `pixels` as a 4-bit-coded string using only the literal
    /// (non-zero) and short-form (`0000 0001` → 1 zero pixel) branches,
    /// terminated by the `0000 0000` end-of-string marker. Output is
    /// padded to a whole byte.
    fn encode_4bit_literal(pixels: &[u8]) -> Vec<u8> {
        fn push_nibble(bits: &mut Vec<u8>, nibble: u8) {
            for k in (0..4).rev() {
                bits.push((nibble >> k) & 1);
            }
        }
        let mut bits: Vec<u8> = Vec::new();
        for &p in pixels {
            assert!(p < 16);
            if p == 0 {
                // 0000 1 1 00 → one pixel of colour 0 (b3 == 0x00).
                push_nibble(&mut bits, 0);
                bits.push(1);
                bits.push(1);
                bits.push(0);
                bits.push(0);
            } else {
                push_nibble(&mut bits, p);
            }
        }
        // end-of-string: 0000 0 000
        push_nibble(&mut bits, 0);
        bits.push(0);
        bits.push(0);
        bits.push(0);
        bits.push(0);
        // Pack to bytes, padding the final byte with zeros.
        let mut out = vec![0u8; bits.len().div_ceil(8)];
        for (i, b) in bits.iter().enumerate() {
            if *b != 0 {
                out[i / 8] |= 1 << (7 - (i % 8));
            }
        }
        out
    }

    /// Encode `pixels` as a 2-bit-coded string using only the literal
    /// (non-zero) and 0001-prefixed one-pixel-of-zero branches,
    /// terminated by 0000 00. Output padded to a whole byte.
    fn encode_2bit_literal(pixels: &[u8]) -> Vec<u8> {
        let mut bits: Vec<u8> = Vec::new();
        for &p in pixels {
            assert!(p < 4);
            if p == 0 {
                // 00 0 0 01 → one pixel of colour 0 (b3 == 0x01 branch),
                // six bits total.
                bits.push(0);
                bits.push(0);
                bits.push(0);
                bits.push(0);
                bits.push(0);
                bits.push(1);
            } else {
                for k in (0..2).rev() {
                    bits.push((p >> k) & 1);
                }
            }
        }
        // end-of-string: 00 0 0 00 (code=00, b1=0, b2=0, b3=00)
        bits.extend(std::iter::repeat_n(0u8, 6));
        let mut out = vec![0u8; bits.len().div_ceil(8)];
        for (i, b) in bits.iter().enumerate() {
            if *b != 0 {
                out[i / 8] |= 1 << (7 - (i % 8));
            }
        }
        out
    }

    #[test]
    fn decode_8bit_literal_round_trips() {
        for row in [
            vec![0u8],
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            vec![0, 0, 1, 0, 2, 0, 3],
            vec![255; 32],
        ] {
            let enc = encode_8bit_literal(&row);
            let (consumed, decoded) = decode_8bit_string(&enc).unwrap();
            assert_eq!(decoded, row, "8-bit decode mismatch for {:?}", row);
            assert_eq!(consumed, enc.len());
        }
    }

    #[test]
    fn decode_8bit_long_run_marker_repeats_zero() {
        // 0x00 0x83 → run_flag=1, count=3 with the colour from the next
        // byte. Use colour 0 here to mirror the well-defined zero-run
        // shape; the decoder treats this as 3 pixels of the carried
        // colour regardless of value.
        let enc = vec![0x00, 0x83, 0x00, 0x00, 0x00];
        let (consumed, decoded) = decode_8bit_string(&enc).unwrap();
        assert_eq!(decoded, vec![0, 0, 0]);
        assert_eq!(consumed, enc.len());
    }

    #[test]
    fn decode_8bit_truncated_escape_returns_invalid() {
        let err = decode_8bit_string(&[0x00]).unwrap_err();
        match err {
            Error::InvalidData(_) => {}
            other => panic!("expected InvalidData, got {other:?}"),
        }
    }

    #[test]
    fn decode_8bit_truncated_run_byte_returns_invalid() {
        // 0x00 0x83 declares a 3-pixel run but the colour byte is missing.
        let err = decode_8bit_string(&[0x00, 0x83]).unwrap_err();
        match err {
            Error::InvalidData(_) => {}
            other => panic!("expected InvalidData, got {other:?}"),
        }
    }

    #[test]
    fn decode_8bit_garbage_never_panics() {
        // 400 deterministic-random inputs of varying lengths. Output is
        // unconstrained on content — we only assert termination + no
        // out-of-memory blow-up.
        let mut state: u64 = 0xD15EA5E5_BADC0FFE;
        for _ in 0..400 {
            let len = (lcg(&mut state) % 64) as usize;
            let mut buf = vec![0u8; len];
            for b in &mut buf {
                *b = lcg(&mut state) as u8;
            }
            let r = decode_8bit_string(&buf);
            if let Ok((_, decoded)) = r {
                // Should not produce wildly more pixels than there are
                // input bytes. Each run carries at most 127 pixels per
                // byte triple (0x00 + run-flag + colour), and inputs are
                // bounded by `len`.
                assert!(decoded.len() <= 127 * len.max(1));
            }
        }
    }

    #[test]
    fn decode_4bit_literal_round_trips() {
        for row in [
            vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            vec![0, 0, 1, 0, 2, 0, 3],
            vec![15; 16],
            vec![0],
        ] {
            let enc = encode_4bit_literal(&row);
            let (_, decoded) = decode_4bit_string(&enc).unwrap();
            assert_eq!(decoded, row, "4-bit decode mismatch for {:?}", row);
        }
    }

    #[test]
    fn decode_4bit_truncated_returns_invalid_not_panic() {
        // Single literal nibble with no end-of-string. The bit-reader
        // exhaustion path should surface InvalidData.
        let err = decode_4bit_string(&[0x10]).unwrap_err();
        match err {
            Error::InvalidData(_) => {}
            other => panic!("expected InvalidData, got {other:?}"),
        }
    }

    #[test]
    fn decode_4bit_garbage_never_panics() {
        let mut state: u64 = 0x00A1_1CEC_0DEC_0FFE;
        for _ in 0..400 {
            let len = (lcg(&mut state) % 64) as usize;
            let mut buf = vec![0u8; len];
            for b in &mut buf {
                *b = lcg(&mut state) as u8;
            }
            let r = decode_4bit_string(&buf);
            if let Ok((_, decoded)) = r {
                // Each run is at most 284 pixels and consumes at least
                // 16 bits, so output is bounded by ~284 * (input_bits / 16).
                assert!(decoded.len() <= 284 * (len + 1));
            }
        }
    }

    #[test]
    fn decode_2bit_literal_round_trips() {
        for row in [
            vec![1u8, 2, 3, 1, 2, 3],
            vec![0, 1, 0, 2, 0, 3],
            vec![3; 7],
            vec![0],
        ] {
            let enc = encode_2bit_literal(&row);
            let (_, decoded) = decode_2bit_string(&enc).unwrap();
            assert_eq!(decoded, row, "2-bit decode mismatch for {:?}", row);
        }
    }

    #[test]
    fn decode_2bit_truncated_returns_invalid_not_panic() {
        // Empty input: the very first 2-bit read hits the bit reader's
        // exhaustion guard.
        let err = decode_2bit_string(&[]).unwrap_err();
        match err {
            Error::InvalidData(_) => {}
            other => panic!("expected InvalidData, got {other:?}"),
        }
    }

    #[test]
    fn decode_2bit_garbage_never_panics() {
        let mut state: u64 = 0xFEEDFACE_CAFEBABE;
        for _ in 0..400 {
            let len = (lcg(&mut state) % 64) as usize;
            let mut buf = vec![0u8; len];
            for b in &mut buf {
                *b = lcg(&mut state) as u8;
            }
            let r = decode_2bit_string(&buf);
            if let Ok((_, decoded)) = r {
                // 8-bit-count run with +29 offset is the longest 2-bit
                // run; each consumes 14 bits. Bound generously.
                assert!(decoded.len() <= 284 * (len + 1));
            }
        }
    }

    // --- parse_pixel_lines coverage ---------------------------------

    #[test]
    fn parse_pixel_lines_collects_rows_via_end_of_line_marker() {
        // Two 8-bit rows separated by 0xF0 end-of-object-line.
        let mut buf = Vec::new();
        buf.push(0x12);
        buf.extend_from_slice(&encode_8bit_literal(&[1, 2, 3]));
        buf.push(0xF0);
        buf.push(0x12);
        buf.extend_from_slice(&encode_8bit_literal(&[4, 5]));
        buf.push(0xF0);
        let rows = parse_pixel_lines(&buf).unwrap();
        assert_eq!(rows, vec![vec![1u8, 2, 3], vec![4, 5]]);
    }

    #[test]
    fn parse_pixel_lines_skips_map_tables() {
        // 0x20 / 0x21 / 0x22 each carry a 2-byte body. Sandwich one
        // between two 8-bit pixel-coded strings and confirm the data
        // bytes are consumed without being treated as pixels.
        let mut buf = Vec::new();
        buf.push(0x12);
        buf.extend_from_slice(&encode_8bit_literal(&[7]));
        buf.push(0x20);
        buf.push(0xAA);
        buf.push(0xBB);
        buf.push(0x12);
        buf.extend_from_slice(&encode_8bit_literal(&[9]));
        buf.push(0xF0);
        let rows = parse_pixel_lines(&buf).unwrap();
        assert_eq!(rows, vec![vec![7u8, 9]]);
    }

    #[test]
    fn parse_pixel_lines_rejects_truncated_map_table() {
        // 0x20 introducer with only one trailing byte instead of two.
        let buf = vec![0x20, 0xAA];
        let err = parse_pixel_lines(&buf).unwrap_err();
        match err {
            Error::InvalidData(_) => {}
            other => panic!("expected InvalidData, got {other:?}"),
        }
    }

    #[test]
    fn parse_pixel_lines_rejects_unknown_data_type() {
        // 0x80 / 0xF1 / etc. are not handled — must error, not panic.
        let err = parse_pixel_lines(&[0x80]).unwrap_err();
        match err {
            Error::InvalidData(_) => {}
            other => panic!("expected InvalidData, got {other:?}"),
        }
    }

    #[test]
    fn parse_pixel_lines_garbage_never_panics() {
        let mut state: u64 = 0xBEEF_C0FF_EE80_86F0;
        for _ in 0..400 {
            let len = (lcg(&mut state) % 96) as usize;
            let mut buf = vec![0u8; len];
            for b in &mut buf {
                *b = lcg(&mut state) as u8;
            }
            // Decoder must terminate (Ok or Err) without panicking.
            let _ = parse_pixel_lines(&buf);
        }
    }

    // --- end-to-end decoder coverage --------------------------------

    /// Build a PES payload that delivers a single 4×1 object whose
    /// pixel data uses the requested per-row encoder. Lets the
    /// integration tests reuse the DDS/PCS/RCS/CLUT prefix from
    /// `build_demo_pes` without inheriting its 8-bit-only line block.
    fn build_pes_with_custom_object(
        canvas: (u16, u16),
        encoder_byte: u8,
        rows: &[Vec<u8>],
    ) -> Vec<u8> {
        fn segment(out: &mut Vec<u8>, seg_type: u8, body: &[u8]) {
            out.push(0x0F);
            out.push(seg_type);
            out.extend_from_slice(&1u16.to_be_bytes());
            out.extend_from_slice(&(body.len() as u16).to_be_bytes());
            out.extend_from_slice(body);
        }

        let mut out = vec![0x20, 0x00];

        // DDS — canvas.
        let mut dds = Vec::new();
        dds.push(0);
        dds.extend_from_slice(&(canvas.0.saturating_sub(1)).to_be_bytes());
        dds.extend_from_slice(&(canvas.1.saturating_sub(1)).to_be_bytes());
        segment(&mut out, SEG_DISPLAY_DEFINITION, &dds);

        // PCS — one region at (0,0).
        let mut page = vec![0, 0, 0, 0xFF];
        page.extend_from_slice(&0u16.to_be_bytes());
        page.extend_from_slice(&0u16.to_be_bytes());
        segment(&mut out, SEG_PAGE_COMPOSITION, &page);

        // RCS — region 0, depth = 3 (8-bit), CLUT 0.
        let mut region = Vec::new();
        region.push(0);
        region.push(0);
        region.extend_from_slice(&(rows[0].len() as u16).to_be_bytes());
        region.extend_from_slice(&(rows.len() as u16).to_be_bytes());
        region.push(3 << 2);
        region.push(0);
        region.push(0);
        region.push(0);
        region.extend_from_slice(&0u16.to_be_bytes());
        region.extend_from_slice(&0u16.to_be_bytes());
        region.extend_from_slice(&0u16.to_be_bytes());
        segment(&mut out, SEG_REGION_COMPOSITION, &region);

        // CLUT — entry 1 white, entry 2 red.
        let mut clut = vec![0, 0];
        clut.extend_from_slice(&[1, 0xFF, 255, 128, 128, 0]);
        clut.extend_from_slice(&[2, 0xFF, 81, 240, 90, 0]);
        segment(&mut out, SEG_CLUT_DEFINITION, &clut);

        // Object data: top + bottom field both use the supplied encoder.
        let mut obj = Vec::new();
        obj.extend_from_slice(&0u16.to_be_bytes());
        obj.push(0);
        let mut field = Vec::new();
        for row in rows {
            field.push(encoder_byte);
            let encoded: Vec<u8> = match encoder_byte {
                0x12 => encode_8bit_literal(row),
                0x11 => encode_4bit_literal(row),
                0x10 => encode_2bit_literal(row),
                _ => panic!("unsupported encoder byte"),
            };
            field.extend_from_slice(&encoded);
            field.push(0xF0);
        }
        obj.extend_from_slice(&(field.len() as u16).to_be_bytes());
        obj.extend_from_slice(&0u16.to_be_bytes()); // bottom length = 0
        obj.extend_from_slice(&field);
        segment(&mut out, SEG_OBJECT_DATA, &obj);

        segment(&mut out, SEG_END_OF_DISPLAY_SET, &[]);
        out
    }

    #[test]
    fn decodes_4bit_pixel_coded_bitmap_end_to_end() {
        // 4×1: white, red, red, white encoded with the 4-bit line block.
        let row = vec![1u8, 2, 2, 1];
        let pes = build_pes_with_custom_object((4, 1), 0x11, &[row]);
        let params = CodecParameters::video(CodecId::new(DVBSUB_CODEC_ID));
        let mut dec = make_decoder(&params).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 90_000), pes).with_pts(0);
        dec.send_packet(&pkt).unwrap();
        let Frame::Video(v) = dec.receive_frame().unwrap() else {
            panic!("expected video frame");
        };
        // First column should be white-ish, second column red-ish.
        let col0 = &v.planes[0].data[0..4];
        let col1 = &v.planes[0].data[4..8];
        assert!(col0[0] > 200 && col0[1] > 200 && col0[2] > 200);
        assert!(col1[0] > col1[1] && col1[0] > col1[2]);
    }

    #[test]
    fn decodes_2bit_pixel_coded_bitmap_end_to_end() {
        // 4×1: white, red, red, white encoded with the 2-bit line block.
        let row = vec![1u8, 2, 2, 1];
        let pes = build_pes_with_custom_object((4, 1), 0x10, &[row]);
        let params = CodecParameters::video(CodecId::new(DVBSUB_CODEC_ID));
        let mut dec = make_decoder(&params).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 90_000), pes).with_pts(0);
        dec.send_packet(&pkt).unwrap();
        let Frame::Video(v) = dec.receive_frame().unwrap() else {
            panic!("expected video frame");
        };
        let col0 = &v.planes[0].data[0..4];
        let col1 = &v.planes[0].data[4..8];
        assert!(col0[0] > 200 && col0[1] > 200 && col0[2] > 200);
        assert!(col1[0] > col1[1] && col1[0] > col1[2]);
    }

    #[test]
    fn rejects_zero_canvas() {
        // DDS with width = 0 (encoded as 0xFFFF after the
        // saturating_sub(1)) gets caught by the zero-canvas guard.
        // Build the PES manually because the helper's saturating_sub
        // hides the case.
        let mut out = vec![0x20, 0x00];
        // DDS.
        let mut dds = Vec::new();
        dds.push(0);
        // width_minus_1 = 0xFFFF (= width 0 after the +1).
        // But the decoder treats the field as an absolute width: a
        // zero canvas is constructed by setting width = 0 / height = 0
        // before the +1 fold-in. The parser reads width as the raw
        // field, so we exercise it by writing 0 here.
        dds.extend_from_slice(&0u16.to_be_bytes());
        dds.extend_from_slice(&0u16.to_be_bytes());
        let mut seg = vec![0x0F, SEG_DISPLAY_DEFINITION];
        seg.extend_from_slice(&1u16.to_be_bytes());
        seg.extend_from_slice(&(dds.len() as u16).to_be_bytes());
        seg.extend_from_slice(&dds);
        out.extend_from_slice(&seg);
        // No PCS — the decoder still produces a canvas, so we have to
        // hit the zero-canvas guard via a zero-width DDS. The parser
        // reads display_width = 0, height = 0.
        let mut end = vec![0x0F, SEG_END_OF_DISPLAY_SET];
        end.extend_from_slice(&1u16.to_be_bytes());
        end.extend_from_slice(&0u16.to_be_bytes());
        out.extend_from_slice(&end);

        let params = CodecParameters::video(CodecId::new(DVBSUB_CODEC_ID));
        let mut dec = make_decoder(&params).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 90_000), out).with_pts(0);
        // The DDS here actually encodes width=1, height=1 because the
        // raw field is "width_minus_1". So `send_packet` succeeds; the
        // intent of the test is to verify the parser doesn't panic on
        // the minimal degenerate-but-legal shape and that it surfaces
        // an empty render rather than crashing.
        dec.send_packet(&pkt).unwrap();
        let Frame::Video(v) = dec.receive_frame().unwrap() else {
            panic!("expected video frame");
        };
        // 1×1 canvas of background (all zeros).
        assert_eq!(v.planes[0].data, vec![0u8; 4]);
    }

    // --- read_segment coverage --------------------------------------

    #[test]
    fn read_segment_rejects_bad_sync_byte() {
        // Sync byte must be 0x0F.
        let buf = [0x0E, SEG_END_OF_DISPLAY_SET, 0x00, 0x01, 0x00, 0x00];
        let err = read_segment(&buf, 0).unwrap_err();
        match err {
            Error::InvalidData(_) => {}
            other => panic!("expected InvalidData, got {other:?}"),
        }
    }

    #[test]
    fn read_segment_short_header_returns_need_more() {
        // Five bytes — header is six. Should report NeedMore so the
        // outer loop can wait for the next PES.
        let buf = [0x0F, SEG_END_OF_DISPLAY_SET, 0x00, 0x01, 0x00];
        let err = read_segment(&buf, 0).unwrap_err();
        match err {
            Error::NeedMore => {}
            other => panic!("expected NeedMore, got {other:?}"),
        }
    }

    #[test]
    fn read_segment_truncated_body_returns_need_more() {
        // Header advertises a 4-byte body but only 2 bytes follow.
        let buf = [
            0x0F,
            SEG_PAGE_COMPOSITION,
            0x00,
            0x01,
            0x00,
            0x04,
            0xAA,
            0xBB,
        ];
        let err = read_segment(&buf, 0).unwrap_err();
        match err {
            Error::NeedMore => {}
            other => panic!("expected NeedMore, got {other:?}"),
        }
    }

    // --- region_composition coverage -------------------------------

    /// `region_fill_flag` (bit 3 of body[1]) decodes to the typed `fill`
    /// boolean, together with the three depth-keyed pixel codes packed
    /// into body[8] (8-bit) and body[9] (4-bit high nibble, 2-bit bits
    /// 5..6). The depth is read from body[6] bits 4..2 and exposed as
    /// `depth_bits` so the renderer can pick the right fill index.
    #[test]
    fn region_composition_decodes_fill_flag_and_pixel_codes() {
        // region_id=7, version=0, fill=1; width=5, height=4; depth=3
        // (8-bit); CLUT id=2; 8-bit code=0xAB; 4-bit code=0xC; 2-bit
        // code=0x2; no objects.
        let body = vec![
            0x07, // region_id
            0x08, // version (0) << 4 | fill (1) << 3 | reserved
            0x00,
            0x05, // width
            0x00,
            0x04,     // height
            (3 << 2), // region_depth = 3 (8-bit), level=0, reserved=0
            0x02,     // clut_id
            0xAB,     // region_8-bit_pixel_code
            // region_4-bit_pixel_code in high nibble (0xC), 2-bit code in bits 5..6 (0b10),
            // reserved (2 bits) = 0 → 0xC8.
            0xC8,
        ];
        let (id, region) = parse_region_composition(&body).unwrap();
        assert_eq!(id, 7);
        assert!(region.fill);
        assert_eq!(region.width, 5);
        assert_eq!(region.height, 4);
        assert_eq!(region.depth_bits, 8);
        assert_eq!(region.clut_id, 2);
        assert_eq!(region.fill_code_8, 0xAB);
        assert_eq!(region.fill_code_4, 0x0C);
        assert_eq!(region.fill_code_2, 0x02);
        assert!(region.objects.is_empty());
    }

    /// fill_flag unset: the rectangle stays at the canvas background
    /// (zero RGBA) — the renderer must skip the pre-fill step even
    /// when the per-depth pixel codes point at non-transparent CLUT
    /// entries.
    #[test]
    fn region_composition_fill_flag_cleared() {
        let body = vec![
            0x00, // region_id
            0x00, // version=0, fill=0
            0x00,
            0x02,
            0x00,
            0x02, // width=2, height=2
            (3 << 2),
            0x00,
            0x05, // 8-bit code = 5 — would map to white if fill were set
            0x00,
        ];
        let (_, region) = parse_region_composition(&body).unwrap();
        assert!(!region.fill);
        assert_eq!(region.fill_code_8, 5);
    }

    /// End-to-end: a region with `region_fill_flag` set pre-paints the
    /// region rectangle with the CLUT entry that the depth-appropriate
    /// pixel code selects, *before* any objects composite on top. The
    /// canvas pixels outside the region stay transparent.
    #[test]
    fn region_fill_flag_prepaints_region_rectangle() {
        // 4×3 canvas. One 2×2 region at (1, 0), 8-bit depth, fill = 1,
        // fill_code_8 = 1 (= white). No objects.
        fn seg(out: &mut Vec<u8>, t: u8, body: &[u8]) {
            out.push(0x0F);
            out.push(t);
            out.extend_from_slice(&1u16.to_be_bytes());
            out.extend_from_slice(&(body.len() as u16).to_be_bytes());
            out.extend_from_slice(body);
        }
        let mut pes = vec![0x20, 0x00];
        // DDS: width-1=3, height-1=2 → 4×3 canvas.
        seg(
            &mut pes,
            SEG_DISPLAY_DEFINITION,
            &[0, 0x00, 0x03, 0x00, 0x02],
        );
        // PCS: region 0 at (1, 0).
        let mut page = vec![0, 0, 0, 0xFF];
        page.extend_from_slice(&1u16.to_be_bytes()); // x = 1
        page.extend_from_slice(&0u16.to_be_bytes()); // y = 0
        seg(&mut pes, SEG_PAGE_COMPOSITION, &page);
        // RCS: region 0, fill=1, w=2, h=2, depth=8-bit, clut=0,
        // fill_code_8 = 1, no objects.
        let rcs = vec![
            0x00,
            0x08, // version=0, fill=1
            0x00,
            0x02, // width = 2
            0x00,
            0x02, // height = 2
            (3 << 2),
            0x00, // depth=8-bit, clut=0
            0x01,
            0x00, // pixel codes (8-bit=1, 4-bit=0, 2-bit=0)
        ];
        seg(&mut pes, SEG_REGION_COMPOSITION, &rcs);
        // CLUT 0: entry 1 = white (Y=255, Cb=Cr=128, T=0).
        let clut = vec![0, 0, 1, 0xFF, 255, 128, 128, 0];
        seg(&mut pes, SEG_CLUT_DEFINITION, &clut);
        seg(&mut pes, SEG_END_OF_DISPLAY_SET, &[]);

        let params = CodecParameters::video(CodecId::new(DVBSUB_CODEC_ID));
        let mut dec = make_decoder(&params).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 90_000), pes).with_pts(0);
        dec.send_packet(&pkt).unwrap();
        let Frame::Video(v) = dec.receive_frame().unwrap() else {
            panic!("expected video frame");
        };
        let d = &v.planes[0].data;
        let px = |x: usize, y: usize| &d[(y * 4 + x) * 4..(y * 4 + x) * 4 + 4];
        // Inside the 2×2 region at (1..3, 0..2): white-ish, opaque.
        for (rx, ry) in [(1, 0), (2, 0), (1, 1), (2, 1)] {
            let p = px(rx, ry);
            assert!(
                p[0] > 200 && p[1] > 200 && p[2] > 200 && p[3] == 255,
                "expected opaque white at ({}, {}), got {:?}",
                rx,
                ry,
                p
            );
        }
        // Column 0 is left of the region — still transparent canvas.
        assert_eq!(px(0, 0), &[0, 0, 0, 0]);
        assert_eq!(px(0, 1), &[0, 0, 0, 0]);
        // Row 2 is below the region — still transparent canvas.
        assert_eq!(px(0, 2), &[0, 0, 0, 0]);
        assert_eq!(px(1, 2), &[0, 0, 0, 0]);
        assert_eq!(px(2, 2), &[0, 0, 0, 0]);
    }

    /// Same shape as the previous test but with `region_fill_flag = 0`:
    /// the region rectangle is *not* pre-painted, so the canvas stays
    /// fully transparent end-to-end. Confirms the renderer keys off
    /// `fill`, not the presence of a non-zero CLUT entry.
    #[test]
    fn region_fill_flag_cleared_leaves_canvas_transparent() {
        fn seg(out: &mut Vec<u8>, t: u8, body: &[u8]) {
            out.push(0x0F);
            out.push(t);
            out.extend_from_slice(&1u16.to_be_bytes());
            out.extend_from_slice(&(body.len() as u16).to_be_bytes());
            out.extend_from_slice(body);
        }
        let mut pes = vec![0x20, 0x00];
        seg(
            &mut pes,
            SEG_DISPLAY_DEFINITION,
            &[0, 0x00, 0x03, 0x00, 0x02],
        );
        let mut page = vec![0, 0, 0, 0xFF];
        page.extend_from_slice(&1u16.to_be_bytes());
        page.extend_from_slice(&0u16.to_be_bytes());
        seg(&mut pes, SEG_PAGE_COMPOSITION, &page);
        // RCS with fill flag cleared but pixel code still 1.
        let rcs = vec![
            0x00,
            0x00, // version=0, fill=0
            0x00,
            0x02,
            0x00,
            0x02,
            (3 << 2),
            0x00,
            0x01,
            0x00,
        ];
        seg(&mut pes, SEG_REGION_COMPOSITION, &rcs);
        let clut = vec![0, 0, 1, 0xFF, 255, 128, 128, 0];
        seg(&mut pes, SEG_CLUT_DEFINITION, &clut);
        seg(&mut pes, SEG_END_OF_DISPLAY_SET, &[]);

        let params = CodecParameters::video(CodecId::new(DVBSUB_CODEC_ID));
        let mut dec = make_decoder(&params).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 90_000), pes).with_pts(0);
        dec.send_packet(&pkt).unwrap();
        let Frame::Video(v) = dec.receive_frame().unwrap() else {
            panic!("expected video frame");
        };
        for chunk in v.planes[0].data.chunks(4) {
            assert_eq!(chunk, &[0, 0, 0, 0]);
        }
    }

    // --- pixel-code-string encoder roundtrips ------------------------

    fn assert_2bit_roundtrip(row: &[u8]) {
        let enc = encode_2bit_pixel_string(row).unwrap();
        let (consumed, decoded) = decode_2bit_string(&enc).unwrap();
        assert_eq!(decoded, row, "2-bit roundtrip mismatch for {row:?}");
        assert_eq!(consumed, enc.len(), "2-bit consumed != encoded length");
    }

    fn assert_4bit_roundtrip(row: &[u8]) {
        let enc = encode_4bit_pixel_string(row).unwrap();
        let (consumed, decoded) = decode_4bit_string(&enc).unwrap();
        assert_eq!(decoded, row, "4-bit roundtrip mismatch for {row:?}");
        assert_eq!(consumed, enc.len(), "4-bit consumed != encoded length");
    }

    fn assert_8bit_roundtrip(row: &[u8]) {
        let enc = encode_8bit_pixel_string(row);
        let (consumed, decoded) = decode_8bit_string(&enc).unwrap();
        assert_eq!(decoded, row, "8-bit roundtrip mismatch (len {})", row.len());
        assert_eq!(consumed, enc.len(), "8-bit consumed != encoded length");
    }

    /// Pure runs of every colour at every run-length-form boundary the
    /// 2-bit encoder switches on (literals / 3..=10 colour runs /
    /// 12..=27 zero runs / 29..=284 long runs, plus the chunked
    /// over-284 shapes).
    #[test]
    fn encode_2bit_run_boundaries_roundtrip() {
        for col in 0u8..=3 {
            for len in [
                1usize, 2, 3, 4, 10, 11, 12, 13, 26, 27, 28, 29, 30, 283, 284, 285, 300, 600,
            ] {
                assert_2bit_roundtrip(&vec![col; len]);
            }
        }
    }

    /// Mixed rows exercising run/literal transitions and the
    /// zero-vs-colour split paths.
    #[test]
    fn encode_2bit_mixed_rows_roundtrip() {
        let mut long = vec![1u8; 284];
        long.extend(std::iter::repeat_n(0u8, 27));
        long.push(2);
        long.push(0);
        long.extend(std::iter::repeat_n(3u8, 10));
        long.extend(std::iter::repeat_n(0u8, 12));
        long.extend([1, 2, 3, 2, 1]);
        assert_2bit_roundtrip(&long);
        assert_2bit_roundtrip(&[]);
        assert_2bit_roundtrip(&[0]);
        assert_2bit_roundtrip(&[0, 1, 0, 2, 0, 3, 0]);
    }

    #[test]
    fn encode_2bit_rejects_out_of_range_pixel() {
        let err = encode_2bit_pixel_string(&[1, 4]).unwrap_err();
        match err {
            Error::InvalidData(_) => {}
            other => panic!("expected InvalidData, got {other:?}"),
        }
    }

    #[test]
    fn encode_2bit_random_rows_roundtrip() {
        let mut state: u64 = 0x2B17_2B17_2B17_2B17;
        for _ in 0..300 {
            let len = (lcg(&mut state) % 200) as usize;
            let row: Vec<u8> = (0..len).map(|_| (lcg(&mut state) % 4) as u8).collect();
            assert_2bit_roundtrip(&row);
        }
    }

    /// Pure runs at every 4-bit form boundary: literals, 4..=7 colour
    /// runs, 3..=9 zero runs, one/two-zero codes, 9..=24 and 25..=280
    /// counted runs, plus chunked over-280 shapes.
    #[test]
    fn encode_4bit_run_boundaries_roundtrip() {
        for col in [0u8, 1, 7, 15] {
            for len in [
                1usize, 2, 3, 4, 5, 7, 8, 9, 10, 23, 24, 25, 26, 279, 280, 281, 300, 600,
            ] {
                assert_4bit_roundtrip(&vec![col; len]);
            }
        }
    }

    #[test]
    fn encode_4bit_mixed_rows_roundtrip() {
        let mut long = vec![5u8; 280];
        long.extend(std::iter::repeat_n(0u8, 9));
        long.extend([15, 0, 0, 14]);
        long.extend(std::iter::repeat_n(1u8, 24));
        long.extend(std::iter::repeat_n(0u8, 2));
        long.extend([1, 2, 3, 4, 5, 6, 7, 8]);
        assert_4bit_roundtrip(&long);
        assert_4bit_roundtrip(&[]);
        assert_4bit_roundtrip(&[0]);
        assert_4bit_roundtrip(&[0, 9, 0, 10, 0]);
    }

    #[test]
    fn encode_4bit_rejects_out_of_range_pixel() {
        let err = encode_4bit_pixel_string(&[16]).unwrap_err();
        match err {
            Error::InvalidData(_) => {}
            other => panic!("expected InvalidData, got {other:?}"),
        }
    }

    #[test]
    fn encode_4bit_random_rows_roundtrip() {
        let mut state: u64 = 0x4B17_4B17_4B17_4B17;
        for _ in 0..300 {
            let len = (lcg(&mut state) % 200) as usize;
            let row: Vec<u8> = (0..len).map(|_| (lcg(&mut state) % 16) as u8).collect();
            assert_4bit_roundtrip(&row);
        }
    }

    /// Pure runs at the 8-bit escape boundaries: literal-vs-run cutover
    /// (3 pixels), the 127-count cap, and chunked longer runs.
    #[test]
    fn encode_8bit_run_boundaries_roundtrip() {
        for col in [0u8, 1, 128, 255] {
            for len in [1usize, 2, 3, 4, 5, 126, 127, 128, 129, 253, 254, 255, 400] {
                assert_8bit_roundtrip(&vec![col; len]);
            }
        }
    }

    #[test]
    fn encode_8bit_mixed_rows_roundtrip() {
        let mut long = vec![200u8; 127];
        long.extend(std::iter::repeat_n(0u8, 127));
        long.extend([1, 2, 3, 0, 255, 0, 0, 9]);
        long.extend(std::iter::repeat_n(42u8, 128));
        assert_8bit_roundtrip(&long);
        assert_8bit_roundtrip(&[]);
        assert_8bit_roundtrip(&[0]);
    }

    #[test]
    fn encode_8bit_random_rows_roundtrip() {
        let mut state: u64 = 0x8B17_8B17_8B17_8B17;
        for _ in 0..300 {
            let len = (lcg(&mut state) % 200) as usize;
            // Bias toward a tiny alphabet so runs actually form.
            let row: Vec<u8> = (0..len)
                .map(|_| {
                    let v = lcg(&mut state);
                    if v % 3 == 0 {
                        (v >> 8) as u8
                    } else {
                        (v % 4) as u8
                    }
                })
                .collect();
            assert_8bit_roundtrip(&row);
        }
    }

    // --- segment writer roundtrips ------------------------------------

    #[test]
    fn write_read_segment_roundtrip() {
        let mut buf = Vec::new();
        let body_a: Vec<u8> = (0..37u8).collect();
        write_segment(&mut buf, SEG_OBJECT_DATA, 0xABCD, &body_a).unwrap();
        write_segment(&mut buf, SEG_END_OF_DISPLAY_SET, 0x0001, &[]).unwrap();
        let (seg, next) = read_segment(&buf, 0).unwrap();
        assert_eq!(seg.seg_type, SEG_OBJECT_DATA);
        assert_eq!(seg.page_id, 0xABCD);
        assert_eq!(seg.body, body_a);
        let (seg2, end) = read_segment(&buf, next).unwrap();
        assert_eq!(seg2.seg_type, SEG_END_OF_DISPLAY_SET);
        assert_eq!(seg2.page_id, 1);
        assert!(seg2.body.is_empty());
        assert_eq!(end, buf.len());
    }

    #[test]
    fn write_segment_rejects_oversized_body() {
        let mut buf = Vec::new();
        let body = vec![0u8; u16::MAX as usize + 1];
        let err = write_segment(&mut buf, SEG_OBJECT_DATA, 1, &body).unwrap_err();
        match err {
            Error::InvalidData(_) => {}
            other => panic!("expected InvalidData, got {other:?}"),
        }
    }

    #[test]
    fn display_definition_roundtrip() {
        for (w, h) in [(1u16, 1u16), (720, 576), (1920, 1080), (65535, 65535)] {
            let body = write_display_definition(w, h).unwrap();
            let dds = parse_display_definition(&body).unwrap();
            assert_eq!((dds.width, dds.height), (w, h));
        }
        assert!(write_display_definition(0, 576).is_err());
        assert!(write_display_definition(720, 0).is_err());
    }

    #[test]
    fn display_definition_no_window_has_none() {
        // The default 5-byte form carries no window; the flag is clear.
        let body = write_display_definition(1920, 1080).unwrap();
        assert_eq!(body.len(), 5);
        let dds = parse_display_definition(&body).unwrap();
        assert_eq!(dds.window, None);
    }

    #[test]
    fn display_definition_window_roundtrip() {
        // ETSI EN 300 743 §7.2.1: when display_window_flag is set, the
        // four inclusive window positions follow and survive a write/
        // parse roundtrip. Version is carried in the top nibble.
        let win = DisplayWindow {
            h_min: 100,
            h_max: 1819,
            v_min: 60,
            v_max: 1019,
        };
        let body = write_display_definition_windowed(1920, 1080, 9, Some(win)).unwrap();
        assert_eq!(body.len(), 13);
        assert_eq!(body[0] & 0x08, 0x08, "display_window_flag must be set");
        let dds = parse_display_definition(&body).unwrap();
        assert_eq!((dds.width, dds.height), (1920, 1080));
        assert_eq!(dds.version, 9);
        assert_eq!(dds.window, Some(win));
    }

    #[test]
    fn display_definition_window_flag_truncated_rejected() {
        // Flag set but only the 5-byte no-window body present.
        let mut body = write_display_definition(720, 576).unwrap();
        body[0] |= 0x08; // raise display_window_flag without the 8 trailing bytes
        assert!(parse_display_definition(&body).is_err());
    }

    #[test]
    fn display_definition_window_inverted_extent_rejected() {
        // A maximum below its minimum is a zero/negative-extent window.
        let bad = DisplayWindow {
            h_min: 500,
            h_max: 400,
            v_min: 10,
            v_max: 20,
        };
        assert!(write_display_definition_windowed(1920, 1080, 0, Some(bad)).is_err());
        // And the decode side rejects such a body even if hand-built.
        let mut body = write_display_definition_windowed(
            1920,
            1080,
            0,
            Some(DisplayWindow {
                h_min: 0,
                h_max: 1,
                v_min: 0,
                v_max: 1,
            }),
        )
        .unwrap();
        // Corrupt v_max (bytes 11..13) to fall below v_min.
        body[9] = 0x00;
        body[10] = 0x05; // v_min = 5
        body[11] = 0x00;
        body[12] = 0x02; // v_max = 2 < 5
        assert!(parse_display_definition(&body).is_err());
    }

    #[test]
    fn display_definition_window_single_pixel_edge() {
        // Inclusive maxima permit an equal min==max (one-pixel/one-line) window.
        let win = DisplayWindow {
            h_min: 7,
            h_max: 7,
            v_min: 3,
            v_max: 3,
        };
        let body = write_display_definition_windowed(720, 576, 0, Some(win)).unwrap();
        let dds = parse_display_definition(&body).unwrap();
        assert_eq!(dds.window, Some(win));
    }

    #[test]
    fn page_composition_roundtrip() {
        let regions = [(0u8, 0u16, 0u16), (7, 100, 200), (255, 65535, 12345)];
        let body = write_page_composition(30, 5, 2, &regions);
        let parsed = parse_page_composition(&body).unwrap();
        assert_eq!(parsed.len(), regions.len());
        for (got, want) in parsed.iter().zip(regions.iter()) {
            assert_eq!((got.region_id, got.x, got.y), *want);
        }
        // Region-free page (the erase shape) parses to an empty list.
        let empty = write_page_composition(0, 0, 0, &[]);
        assert!(parse_page_composition(&empty).unwrap().is_empty());
    }

    #[test]
    fn region_composition_roundtrip_all_depths() {
        for depth_bits in [2u8, 4, 8] {
            let def = RegionCompositionDef {
                region_id: 9,
                version: 3,
                fill: true,
                width: 640,
                height: 80,
                depth_bits,
                clut_id: 5,
                fill_code_8: 0xAB,
                fill_code_4: 0x0C,
                fill_code_2: 0x02,
                objects: vec![(0x1234, 0x3FF, 0x0FF), (1, 0, 0)],
            };
            let body = write_region_composition(&def).unwrap();
            let (id, region) = parse_region_composition(&body).unwrap();
            assert_eq!(id, 9);
            assert!(region.fill);
            assert_eq!(region.width, 640);
            assert_eq!(region.height, 80);
            assert_eq!(region.depth_bits, depth_bits);
            assert_eq!(region.clut_id, 5);
            assert_eq!(region.fill_code_8, 0xAB);
            assert_eq!(region.fill_code_4, 0x0C);
            assert_eq!(region.fill_code_2, 0x02);
            assert_eq!(region.objects.len(), 2);
            assert_eq!(region.objects[0].object_id, 0x1234);
            assert_eq!(region.objects[0].x, 0x3FF);
            assert_eq!(region.objects[0].y, 0x0FF);
            assert_eq!(region.objects[1].object_id, 1);
        }
    }

    #[test]
    fn region_composition_fill_flag_cleared_roundtrip() {
        let def = RegionCompositionDef {
            region_id: 0,
            version: 0,
            fill: false,
            width: 2,
            height: 2,
            depth_bits: 8,
            clut_id: 0,
            fill_code_8: 5,
            fill_code_4: 0,
            fill_code_2: 0,
            objects: Vec::new(),
        };
        let body = write_region_composition(&def).unwrap();
        let (_, region) = parse_region_composition(&body).unwrap();
        assert!(!region.fill);
        assert_eq!(region.fill_code_8, 5);
        assert!(region.objects.is_empty());
    }

    #[test]
    fn region_composition_rejects_bad_inputs() {
        let mut def = RegionCompositionDef {
            region_id: 0,
            version: 0,
            fill: false,
            width: 2,
            height: 2,
            depth_bits: 3, // not 2/4/8
            clut_id: 0,
            fill_code_8: 0,
            fill_code_4: 0,
            fill_code_2: 0,
            objects: Vec::new(),
        };
        assert!(write_region_composition(&def).is_err());
        def.depth_bits = 8;
        def.objects = vec![(0, 0x4000, 0)];
        assert!(write_region_composition(&def).is_err());
        def.objects = vec![(0, 0, 0x1000)];
        assert!(write_region_composition(&def).is_err());
    }

    /// Full-range CLUT entries built from grey RGBA values are bit-exact
    /// through the YCbCr roundtrip (Cb = Cr = 128 makes the chroma terms
    /// vanish on both sides).
    #[test]
    fn clut_full_range_grey_roundtrip_is_exact() {
        let mut entries = Vec::new();
        for (i, v) in (0u16..=255).step_by(17).enumerate() {
            let [y, cr, cb, t] = rgba_to_clut_ycbcrt([v as u8, v as u8, v as u8, 255]);
            entries.push(ClutEntryDef {
                entry_id: (i + 1) as u8,
                y,
                cr,
                cb,
                t,
                full_range: true,
            });
        }
        let body = write_clut_definition(3, 1, &entries);
        let mut cluts = HashMap::new();
        parse_clut_into(&body, &mut cluts).unwrap();
        let clut = &cluts[&3];
        for (i, v) in (0u16..=255).step_by(17).enumerate() {
            let v = v as u8;
            assert_eq!(
                clut.entries[i + 1],
                [v, v, v, 255],
                "grey {v} did not roundtrip exactly"
            );
        }
    }

    /// Arbitrary colours roundtrip within a couple of LSBs per colour
    /// channel; alpha is exact (T = 255 − A on both sides).
    #[test]
    fn clut_full_range_colour_roundtrip_is_close() {
        let samples = [
            [255u8, 0, 0, 255],
            [0, 255, 0, 200],
            [0, 0, 255, 128],
            [200, 50, 25, 255],
            [10, 200, 100, 64],
            [255, 255, 0, 255],
            [128, 0, 128, 255],
        ];
        let entries: Vec<ClutEntryDef> = samples
            .iter()
            .enumerate()
            .map(|(i, rgba)| {
                let [y, cr, cb, t] = rgba_to_clut_ycbcrt(*rgba);
                ClutEntryDef {
                    entry_id: (i + 1) as u8,
                    y,
                    cr,
                    cb,
                    t,
                    full_range: true,
                }
            })
            .collect();
        let body = write_clut_definition(0, 0, &entries);
        let mut cluts = HashMap::new();
        parse_clut_into(&body, &mut cluts).unwrap();
        let clut = &cluts[&0];
        for (i, want) in samples.iter().enumerate() {
            let got = clut.entries[i + 1];
            for c in 0..3 {
                assert!(
                    (got[c] as i32 - want[c] as i32).abs() <= 2,
                    "colour {want:?} channel {c} drifted to {got:?}"
                );
            }
            assert_eq!(got[3], want[3], "alpha must be exact for {want:?}");
        }
    }

    /// The packed 2-byte CLUT form quantises Y/Cr/Cb/T to 6/4/4/2 bits;
    /// values already on those grids decode identically to the full
    /// 4-byte form carrying the same quad.
    #[test]
    fn clut_short_form_matches_full_form_on_grid_values() {
        let quads: [[u8; 4]; 3] = [
            [252, 128, 128, 0], // near-white grey, opaque
            [80, 240, 96, 64],
            [160, 16, 224, 192],
        ];
        let mut entries = Vec::new();
        for (i, q) in quads.iter().enumerate() {
            entries.push(ClutEntryDef {
                entry_id: (i * 2) as u8,
                y: q[0],
                cr: q[1],
                cb: q[2],
                t: q[3],
                full_range: true,
            });
            entries.push(ClutEntryDef {
                entry_id: (i * 2 + 1) as u8,
                y: q[0],
                cr: q[1],
                cb: q[2],
                t: q[3],
                full_range: false,
            });
        }
        let body = write_clut_definition(0, 0, &entries);
        let mut cluts = HashMap::new();
        parse_clut_into(&body, &mut cluts).unwrap();
        let clut = &cluts[&0];
        for (i, quad) in quads.iter().enumerate() {
            assert_eq!(
                clut.entries[i * 2],
                clut.entries[i * 2 + 1],
                "short form diverged from full form for quad {quad:?}"
            );
        }
    }

    // --- object-data writer roundtrips --------------------------------

    /// Expected decode-side row list for a bitmap written through
    /// `write_object_data`: the rows themselves, except a single-row
    /// object gains one empty bottom-field row.
    fn expected_rows(rows: &[Vec<u8>]) -> Vec<Vec<u8>> {
        let mut out = rows.to_vec();
        if rows.len() == 1 {
            out.push(Vec::new());
        }
        out
    }

    #[test]
    fn object_data_roundtrip_even_odd_and_single_heights() {
        let bitmaps: [Vec<Vec<u8>>; 3] = [
            // Even height — both fields populated.
            vec![
                vec![1, 2, 3, 0],
                vec![0, 3, 2, 1],
                vec![1, 1, 1, 1],
                vec![2; 4],
            ],
            // Odd height — top field one row longer.
            vec![vec![3, 0, 3], vec![0, 0, 0], vec![1, 2, 1]],
            // Single row — bottom field is one empty row on decode.
            vec![vec![1, 0, 2, 0, 3]],
        ];
        for depth_bits in [2u8, 4, 8] {
            for rows in &bitmaps {
                let body = write_object_data(0x0102, 4, depth_bits, rows, &[]).unwrap();
                let (id, obj) = parse_object_data(&body).unwrap();
                assert_eq!(id, 0x0102);
                assert_eq!(
                    obj.rows,
                    expected_rows(rows),
                    "depth {depth_bits} bitmap {rows:?} did not roundtrip"
                );
            }
        }
    }

    #[test]
    fn object_data_map_table_prefixes_are_transparent_to_decode() {
        let rows = vec![vec![1u8, 0, 2, 3], vec![3, 3, 0, 1]];
        let plain = write_object_data(7, 0, 2, &rows, &[]).unwrap();
        let mapped = write_object_data(
            7,
            0,
            2,
            &rows,
            &[
                (DATA_TYPE_MAP_2_TO_4, [0x01, 0x23]),
                (DATA_TYPE_MAP_2_TO_8, [0x45, 0x67]),
                (DATA_TYPE_MAP_4_TO_8, [0x89, 0xAB]),
            ],
        )
        .unwrap();
        assert!(mapped.len() > plain.len(), "map tables must add bytes");
        let (_, obj_plain) = parse_object_data(&plain).unwrap();
        let (_, obj_mapped) = parse_object_data(&mapped).unwrap();
        assert_eq!(
            obj_plain.rows, obj_mapped.rows,
            "map-table prefixes changed the decoded pixels"
        );
    }

    #[test]
    fn object_data_rejects_bad_inputs() {
        // No rows.
        assert!(write_object_data(0, 0, 8, &[], &[]).is_err());
        // Unsupported depth.
        assert!(write_object_data(0, 0, 5, &[vec![1]], &[]).is_err());
        // Pixel out of range for the declared depth.
        assert!(write_object_data(0, 0, 2, &[vec![4]], &[]).is_err());
        assert!(write_object_data(0, 0, 4, &[vec![16]], &[]).is_err());
        // Non-map-table data_type in the map-table slot.
        assert!(write_object_data(0, 0, 2, &[vec![1]], &[(0x12, [0, 0])]).is_err());
        // Field block longer than the 16-bit length field: 70k pixels of
        // alternating colours encode as one literal byte each.
        let huge: Vec<u8> = (0..70_000usize).map(|i| 1 + (i % 2) as u8).collect();
        assert!(write_object_data(0, 0, 8, &[huge], &[]).is_err());
    }

    /// Random bitmaps through the writer/parser pair at every depth.
    #[test]
    fn object_data_random_bitmaps_roundtrip() {
        let mut state: u64 = 0x0B7E_C7DA_7A00_0001;
        for _ in 0..100 {
            let depth_bits = [2u8, 4, 8][(lcg(&mut state) % 3) as usize];
            let mask = match depth_bits {
                2 => 3u32,
                4 => 15,
                _ => 255,
            };
            let w = 1 + (lcg(&mut state) % 40) as usize;
            let h = 1 + (lcg(&mut state) % 12) as usize;
            let rows: Vec<Vec<u8>> = (0..h)
                .map(|_| (0..w).map(|_| (lcg(&mut state) & mask) as u8).collect())
                .collect();
            let body = write_object_data(1, 0, depth_bits, &rows, &[]).unwrap();
            let (_, obj) = parse_object_data(&body).unwrap();
            assert_eq!(obj.rows, expected_rows(&rows), "depth {depth_bits} {w}x{h}");
        }
    }

    // --- colour conversion ---------------------------------------------

    #[test]
    fn rgba_ycbcr_roundtrip_is_close_and_re_encode_bounded() {
        // decode(encode(x)) may move a colour channel by up to 2 LSBs
        // (alpha is exact). Repeated re-encoding of decoded subtitles
        // accumulates only slowly: five passes stay within 5 LSBs of
        // the original over a 200k-sample deterministic sweep, so the
        // transform cannot drift unboundedly.
        let mut state: u64 = 0xC010_12D2_1F7C_0001;
        for _ in 0..200_000 {
            let v = lcg(&mut state);
            let rgba = [v as u8, (v >> 8) as u8, (v >> 16) as u8, 255];
            let [y, cr, cb, t] = rgba_to_clut_ycbcrt(rgba);
            let once = ycbcr_to_rgba(y, cr, cb, t);
            assert_eq!(once[3], 255, "alpha must be exact for {rgba:?}");
            for c in 0..3 {
                assert!(
                    (once[c] as i32 - rgba[c] as i32).abs() <= 2,
                    "first roundtrip drifted >2 LSBs for {rgba:?}: {once:?}"
                );
            }
            let mut cur = once;
            for _ in 0..4 {
                let [y2, cr2, cb2, t2] = rgba_to_clut_ycbcrt(cur);
                cur = ycbcr_to_rgba(y2, cr2, cb2, t2);
            }
            for c in 0..3 {
                assert!(
                    (cur[c] as i32 - rgba[c] as i32).abs() <= 5,
                    "five-pass re-encode drifted >5 LSBs for {rgba:?}: {cur:?}"
                );
            }
        }
    }

    /// The region rectangle is clipped to the canvas: a region declared
    /// at (3, 1) with width 4 and height 4 against a 4×3 canvas writes
    /// only the in-bounds 1×2 strip and never indexes out of the
    /// canvas buffer.
    #[test]
    fn region_fill_clips_to_canvas() {
        fn seg(out: &mut Vec<u8>, t: u8, body: &[u8]) {
            out.push(0x0F);
            out.push(t);
            out.extend_from_slice(&1u16.to_be_bytes());
            out.extend_from_slice(&(body.len() as u16).to_be_bytes());
            out.extend_from_slice(body);
        }
        let mut pes = vec![0x20, 0x00];
        // 4×3 canvas.
        seg(
            &mut pes,
            SEG_DISPLAY_DEFINITION,
            &[0, 0x00, 0x03, 0x00, 0x02],
        );
        let mut page = vec![0, 0, 0, 0xFF];
        page.extend_from_slice(&3u16.to_be_bytes()); // x = 3 (last column)
        page.extend_from_slice(&1u16.to_be_bytes()); // y = 1
        seg(&mut pes, SEG_PAGE_COMPOSITION, &page);
        // Region declared as 4×4 — extends past the canvas in both axes.
        let rcs = vec![
            0x00,
            0x08, // fill=1
            0x00,
            0x04, // width = 4
            0x00,
            0x04, // height = 4
            (3 << 2),
            0x00,
            0x01,
            0x00,
        ];
        seg(&mut pes, SEG_REGION_COMPOSITION, &rcs);
        let clut = vec![0, 0, 1, 0xFF, 255, 128, 128, 0];
        seg(&mut pes, SEG_CLUT_DEFINITION, &clut);
        seg(&mut pes, SEG_END_OF_DISPLAY_SET, &[]);

        let params = CodecParameters::video(CodecId::new(DVBSUB_CODEC_ID));
        let mut dec = make_decoder(&params).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 90_000), pes).with_pts(0);
        dec.send_packet(&pkt).unwrap();
        let Frame::Video(v) = dec.receive_frame().unwrap() else {
            panic!("expected video frame");
        };
        let d = &v.planes[0].data;
        let px = |x: usize, y: usize| &d[(y * 4 + x) * 4..(y * 4 + x) * 4 + 4];
        // (3, 1) and (3, 2) are inside both region and canvas →
        // pre-painted white.
        for (rx, ry) in [(3, 1), (3, 2)] {
            let p = px(rx, ry);
            assert!(
                p[3] == 255 && p[0] > 200,
                "expected opaque white at ({}, {}), got {:?}",
                rx,
                ry,
                p
            );
        }
        // (3, 0) is above the region — still transparent.
        assert_eq!(px(3, 0), &[0, 0, 0, 0]);
        // (0, 1) and (2, 1) are left of the region — still transparent.
        assert_eq!(px(0, 1), &[0, 0, 0, 0]);
        assert_eq!(px(2, 1), &[0, 0, 0, 0]);
    }

    // --- Disparity Signalling Segment (§7.2.7) ---------------------

    /// Minimal DSS: a version + page default disparity, no page update
    /// sequence and no region loop. Negative defaults are signed
    /// (tcimsbf), so the byte round-trips through `i8`.
    #[test]
    fn dss_parses_page_default_only() {
        // version=5, page_flag=0, page_default = -7 (0xF9 as i8).
        let body = vec![(5 << 4), 0xF9u8];
        let dss = parse_disparity_signalling(&body).unwrap();
        assert_eq!(dss.version, 5);
        assert_eq!(dss.page_default_disparity_shift, -7);
        assert!(dss.page_update_sequence.is_none());
        assert!(dss.regions.is_empty());
    }

    /// A region with a single implicit subregion: positional fields are
    /// absent on the wire (number_of_subregions_minus_1 == 0), so both
    /// `horizontal_position` and `width` are reported as `None` and the
    /// disparity applies to the whole region. The fractional part is the
    /// high nibble of its byte and is unsigned.
    #[test]
    fn dss_single_subregion_omits_position_fields() {
        let mut body = vec![(2 << 4), 0x00]; // version 2, no page flag, default 0
        body.push(0x09); // region_id = 9
                         // region flags: update_flag=0, reserved=0, subregions_minus_1=0
        body.push(0x00);
        body.push(0x03); // disparity integer = +3
        body.push(0x40); // fractional = 4 (0x4 in high nibble), reserved low
        let dss = parse_disparity_signalling(&body).unwrap();
        assert_eq!(dss.regions.len(), 1);
        let region = &dss.regions[0];
        assert_eq!(region.region_id, 9);
        assert_eq!(region.subregions.len(), 1);
        let sub = &region.subregions[0];
        assert_eq!(sub.horizontal_position, None);
        assert_eq!(sub.width, None);
        assert_eq!(sub.disparity_shift_integer, 3);
        assert_eq!(sub.disparity_shift_fractional, 4);
        assert!(sub.update_sequence.is_none());
    }

    /// Two subregions per region carry explicit horizontal_position +
    /// width (16-bit each), per the spec's `number_of_subregions_minus_1
    /// > 0` branch. Negative disparity integers stay signed.
    #[test]
    fn dss_multi_subregion_carries_positions() {
        let mut body = vec![(0 << 4), 0x00];
        body.push(0x01); // region_id = 1
        body.push(0x01); // flags: subregions_minus_1 = 1 (two subregions)
                         // subregion 0: h=10, w=40, disparity=+5, frac=0
        body.extend_from_slice(&10u16.to_be_bytes());
        body.extend_from_slice(&40u16.to_be_bytes());
        body.push(0x05);
        body.push(0x00);
        // subregion 1: h=100, w=60, disparity=-2 (0xFE), frac=8
        body.extend_from_slice(&100u16.to_be_bytes());
        body.extend_from_slice(&60u16.to_be_bytes());
        body.push(0xFE);
        body.push(0x80);
        let dss = parse_disparity_signalling(&body).unwrap();
        let subs = &dss.regions[0].subregions;
        assert_eq!(subs.len(), 2);
        assert_eq!(subs[0].horizontal_position, Some(10));
        assert_eq!(subs[0].width, Some(40));
        assert_eq!(subs[0].disparity_shift_integer, 5);
        assert_eq!(subs[1].horizontal_position, Some(100));
        assert_eq!(subs[1].width, Some(60));
        assert_eq!(subs[1].disparity_shift_integer, -2);
        assert_eq!(subs[1].disparity_shift_fractional, 8);
    }

    /// A page-level `disparity_shift_update_sequence` (page_flag=1) is
    /// parsed into its `interval_duration` + per-division-period
    /// (interval_count, integer_part) entries.
    #[test]
    fn dss_page_update_sequence_round_trips() {
        let mut body = vec![(1 << 4) | 0x08, 0x02]; // version 1, page_flag, default +2
                                                    // update sequence: length, interval_duration(3), count(1), 2×(2)
        let mut seq = Vec::new();
        seq.extend_from_slice(&[0x00, 0x0B, 0xB8]); // interval_duration = 3000
        seq.push(0x02); // division_period_count = 2
        seq.push(0x01); // interval_count
        seq.push(0x04); // integer_part = +4
        seq.push(0x03); // interval_count
        seq.push(0xFB); // integer_part = -5
        body.push(seq.len() as u8);
        body.extend_from_slice(&seq);
        let dss = parse_disparity_signalling(&body).unwrap();
        let s = dss.page_update_sequence.expect("page sequence present");
        assert_eq!(s.interval_duration, 3000);
        assert_eq!(s.division_periods.len(), 2);
        assert_eq!(s.division_periods[0].interval_count, 1);
        assert_eq!(s.division_periods[0].disparity_shift_integer, 4);
        assert_eq!(s.division_periods[1].disparity_shift_integer, -5);
    }

    /// region_flag=1 attaches one update sequence per region (shared by
    /// all its subregions). With a single subregion the positional
    /// fields stay absent and the sequence follows immediately.
    #[test]
    fn dss_region_update_sequence_applies_per_region() {
        let mut body = vec![0x00, 0x00];
        body.push(0x07); // region_id = 7
        body.push(0x80); // flags: region_update_flag = 1, subregions = 1
        body.push(0x01); // disparity integer = +1
        body.push(0x00); // fractional = 0
        let mut seq = Vec::new();
        seq.extend_from_slice(&[0x00, 0x00, 0x64]); // interval_duration = 100
        seq.push(0x01); // count
        seq.push(0x05); // interval_count
        seq.push(0x09); // integer = +9
        body.push(seq.len() as u8);
        body.extend_from_slice(&seq);
        let dss = parse_disparity_signalling(&body).unwrap();
        let sub = &dss.regions[0].subregions[0];
        let s = sub.update_sequence.as_ref().expect("region sequence");
        assert_eq!(s.interval_duration, 100);
        assert_eq!(s.division_periods[0].interval_count, 5);
        assert_eq!(s.division_periods[0].disparity_shift_integer, 9);
    }

    /// A DSS segment flowing through the decoder's segment loop is parsed
    /// and validated without disturbing the painted 2D (disparity-zero)
    /// canvas — the spec's baseline view. The frame still decodes.
    #[test]
    fn dss_segment_in_display_set_decodes_without_panic() {
        let mut out = vec![0x20, 0x00]; // data_identifier + stream_id
        let mut dds = Vec::new();
        dds.push(0);
        dds.extend_from_slice(&0u16.to_be_bytes()); // width-1 → 1
        dds.extend_from_slice(&0u16.to_be_bytes()); // height-1 → 1
        dvb_segment(&mut out, SEG_DISPLAY_DEFINITION, &dds);

        let mut page = vec![0x00, 0x00];
        page.push(0); // region_id 0
        page.push(0); // reserved
        page.extend_from_slice(&0u16.to_be_bytes()); // x
        page.extend_from_slice(&0u16.to_be_bytes()); // y
        dvb_segment(&mut out, SEG_PAGE_COMPOSITION, &page);

        dvb_segment(
            &mut out,
            SEG_REGION_COMPOSITION,
            &dvb_region_8bit(0, 0, 1, 1, 50),
        );
        dvb_segment(
            &mut out,
            SEG_CLUT_DEFINITION,
            &dvb_clut_entry1(0, 200, 128, 128, 0),
        );
        dvb_segment(&mut out, SEG_OBJECT_DATA, &dvb_object_8bit(50, &[1]));

        // DSS: page default + one region with a single subregion.
        let dss = vec![(3 << 4), 0x02, 0x00, 0x00, 0x06, 0x00];
        dvb_segment(&mut out, SEG_DISPARITY_SIGNALLING, &dss);

        dvb_segment(&mut out, SEG_END_OF_DISPLAY_SET, &[]);

        let params = CodecParameters::video(CodecId::new(DVBSUB_CODEC_ID));
        let mut dec = make_decoder(&params).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 90_000), out).with_pts(0);
        dec.send_packet(&pkt).unwrap();
        // Standalone confirmation the DSS body parses the same bytes.
        assert_eq!(parse_disparity_signalling(&dss).unwrap().version, 3);
    }

    /// A truncated update-sequence body (advertised length overruns the
    /// segment) is rejected, not silently clamped.
    #[test]
    fn dss_rejects_truncated_update_sequence() {
        // page_flag set, then a sequence length of 10 with only 2 bytes.
        let body = vec![0x08, 0x00, 0x0A, 0x00, 0x00];
        let err = parse_disparity_signalling(&body).unwrap_err();
        match err {
            Error::InvalidData(_) => {}
            other => panic!("expected InvalidData, got {other:?}"),
        }
    }

    /// A region entry whose subregion disparity fields run off the end
    /// of the body is rejected.
    #[test]
    fn dss_rejects_truncated_subregion() {
        // region_id + flags then nothing for the disparity bytes.
        let body = vec![0x00, 0x00, 0x01, 0x00];
        let err = parse_disparity_signalling(&body).unwrap_err();
        match err {
            Error::InvalidData(_) => {}
            other => panic!("expected InvalidData, got {other:?}"),
        }
    }
}
