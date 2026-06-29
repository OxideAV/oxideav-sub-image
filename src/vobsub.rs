//! VobSub / DVD SPU parser, container (`.idx`+`.sub`), and decoder.
//!
//! A VobSub title is a text `.idx` file alongside a binary `.sub`
//! file. The `.idx` carries:
//!
//! * a 16-colour YCrCb palette;
//! * the subpicture canvas size;
//! * per-language entries with `timestamp: 00:00:01:000, filepos:
//!   000000000` cue timestamps.
//!
//! The `.sub` file is a tiny MPEG Program Stream (`0x00 00 01 BA` pack
//! headers, `0x00 00 01 BD` private-stream-1 PES packets). Each PES
//! payload carries one SPU unit:
//!
//! ```text
//! SPU size (2 BE)
//! control-block offset (2 BE)  [within the SPU]
//! RLE bitmap bytes             [from 4 to control_offset]
//! control sequences            [from control_offset to SPU size]
//! ```
//!
//! Control sequences consist of:
//!
//! ```text
//! delay (2 BE, in 1024/90000 s units)
//! next-offset (2 BE)
//! command bytes until 0xFF terminator
//! ```
//!
//! Commands:
//!
//! | 0x00 | FSTA_DSP — force start display (no args) — sets [`Spu::forced_display`] and latches [`Spu::start_delay_raw`] from its DCSQ's STM if no earlier start command has been seen |
//! | 0x01 | STA_DSP — start display (no args) — latches [`Spu::start_delay_raw`] from its DCSQ's STM if no earlier start command has been seen |
//! | 0x02 | STP_DSP — stop display (no args) — overwrites [`Spu::stop_delay_raw`] with its DCSQ's STM |
//! | 0x03 | palette sel   | 2 bytes: (bg<<4|pat, emp2<<4|emp1) |
//! | 0x04 | alpha         | 2 bytes: (bg<<4|pat, emp2<<4|emp1) |
//! | 0x05 | coords        | 6 bytes: x1:12 x2:12 y1:12 y2:12 |
//! | 0x06 | rle offsets   | 4 bytes: top_off:16 bot_off:16 |
//! | 0xFF | end           |
//!
//! ## Scope / limitations
//!
//! * **Decode only.**
//! * Handles the standard 4-colour palette + alpha form (every SPU
//!   uses exactly 4 of the 16 palette entries).
//! * The mid-display colour/contrast change command (0x07 CHG_COLCON)
//!   is parsed into a list of vertically-bounded `LN_CTLI` bands, each
//!   carrying one or more `PX_CTLI` (start-column + replacement
//!   palette/alpha) entries, and **applied** to the rendered bitmap
//!   during decoder canvas paint: pixels that fall inside a band's
//!   `csln..=ctln` line range and on/after a `PX_CTLI`'s start-column
//!   pick up that entry's replacement palette and alpha (running until
//!   the next `PX_CTLI` in the same band or the right edge of the
//!   display area). Bands and entries are clipped to the SPU's
//!   declared bounding box (`x1..=x2`, `y1..=y2`); out-of-bbox bands
//!   are ignored. The `Spu::saw_chg_colcon` flag still surfaces that
//!   the command was present, and `Spu::chg_colcon` exposes the parsed
//!   structure for callers that need it.
//! * Palette/alpha defaults are black-text-on-transparent when the
//!   SPU omits a colour command (malformed streams).
//! * `.idx` without a palette line falls back to an all-grey fallback
//!   so tests without a full index still render something.

use std::collections::VecDeque;
use std::io::{Read, SeekFrom};
use std::path::{Path, PathBuf};

use oxideav_core::Decoder;
use oxideav_core::{
    CodecId, CodecParameters, CodecResolver, Error, Frame, MediaType, Packet, PixelFormat, Result,
    StreamInfo, TimeBase, VideoFrame, VideoPlane,
};
use oxideav_core::{ContainerRegistry, Demuxer, ProbeData, ProbeScore, ReadSeek};

use crate::VOBSUB_CODEC_ID;

// --- .idx parser -------------------------------------------------------

/// Parsed contents of a VobSub `.idx` file.
#[derive(Clone, Debug, Default)]
pub struct VobSubIdx {
    pub size: (u16, u16),
    /// 16 entries, each RGB (pre-YCbCr-to-RGB-converted).
    pub palette_rgb: [[u8; 3]; 16],
    pub has_palette: bool,
    /// Cue entries: (start_us, filepos).
    pub cues: Vec<(i64, u64)>,
}

pub fn parse_idx(text: &str) -> Result<VobSubIdx> {
    let mut idx = VobSubIdx::default();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(rest) = line.strip_prefix("size:") {
            let rest = rest.trim();
            if let Some((w, h)) = rest.split_once('x') {
                let w: u16 = w
                    .trim()
                    .parse()
                    .map_err(|e| Error::invalid(format!("vobsub idx: bad size width: {e}")))?;
                let h: u16 = h
                    .trim()
                    .parse()
                    .map_err(|e| Error::invalid(format!("vobsub idx: bad size height: {e}")))?;
                idx.size = (w, h);
            }
        } else if let Some(rest) = line.strip_prefix("palette:") {
            parse_palette_line(rest.trim(), &mut idx)?;
        } else if let Some(rest) = line.strip_prefix("timestamp:") {
            parse_timestamp_line(rest.trim(), &mut idx)?;
        }
    }
    Ok(idx)
}

fn parse_palette_line(s: &str, idx: &mut VobSubIdx) -> Result<()> {
    let mut cnt = 0usize;
    for token in s
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|t| !t.is_empty())
    {
        if cnt >= 16 {
            break;
        }
        let hex = token.trim_start_matches("0x");
        let v = u32::from_str_radix(hex, 16)
            .map_err(|e| Error::invalid(format!("vobsub idx: bad palette entry '{token}': {e}")))?;
        let r = ((v >> 16) & 0xFF) as u8;
        let g = ((v >> 8) & 0xFF) as u8;
        let b = (v & 0xFF) as u8;
        idx.palette_rgb[cnt] = [r, g, b];
        cnt += 1;
    }
    if cnt > 0 {
        idx.has_palette = true;
    }
    Ok(())
}

fn parse_timestamp_line(s: &str, idx: &mut VobSubIdx) -> Result<()> {
    // Expected form: "00:00:01:000, filepos: 000000000"
    let mut ts_str: Option<&str> = None;
    let mut filepos_str: Option<&str> = None;
    for part in s.split(',') {
        let part = part.trim();
        if let Some(rest) = part.strip_prefix("filepos:") {
            filepos_str = Some(rest.trim());
        } else if ts_str.is_none() {
            ts_str = Some(part);
        }
    }
    let ts = ts_str.ok_or_else(|| Error::invalid("vobsub idx: timestamp missing"))?;
    let fp = filepos_str.ok_or_else(|| Error::invalid("vobsub idx: filepos missing"))?;
    let mut parts = ts.split(':');
    let h: i64 = parts
        .next()
        .unwrap_or("0")
        .parse()
        .map_err(|_| Error::invalid("vobsub idx: timestamp hours"))?;
    let m: i64 = parts
        .next()
        .unwrap_or("0")
        .parse()
        .map_err(|_| Error::invalid("vobsub idx: timestamp minutes"))?;
    let s_: i64 = parts
        .next()
        .unwrap_or("0")
        .parse()
        .map_err(|_| Error::invalid("vobsub idx: timestamp seconds"))?;
    let ms: i64 = parts
        .next()
        .unwrap_or("0")
        .parse()
        .map_err(|_| Error::invalid("vobsub idx: timestamp millis"))?;
    let us = ((((h * 60) + m) * 60) + s_) * 1_000_000 + ms * 1_000;
    let filepos = u64::from_str_radix(fp.trim_start_matches("0x"), 16)
        .or_else(|_| fp.parse::<u64>())
        .map_err(|_| Error::invalid("vobsub idx: bad filepos"))?;
    idx.cues.push((us, filepos));
    Ok(())
}

// --- SPU parse + decode -----------------------------------------------

#[derive(Clone, Debug, Default)]
pub struct Spu {
    pub x1: u16,
    pub y1: u16,
    pub x2: u16,
    pub y2: u16,
    /// palette indices (bg, pat, emp1, emp2) into the 16-entry idx palette.
    pub palette_sel: [u8; 4],
    /// alpha values (bg, pat, emp1, emp2), 0..15.
    pub alpha: [u8; 4],
    /// start-display delay in 1024/90000 s units from start of SPU.
    ///
    /// The delay is taken from the `SP_DCSQ_STM` of the Display Control
    /// Sequence that actually carries the `STA_DSP` (0x01) or
    /// `FSTA_DSP` (0x00) command — *not* from the very first DCSQ.
    /// In typical streams the first DCSQ sets palette / alpha / coords
    /// / pixel-data addresses with `STM = 0` and a *later* DCSQ then
    /// schedules the on-display event with a non-zero `STM`; locking
    /// onto the first DCSQ's STM would have reported a permanently
    /// zero start time in that common shape.
    pub start_delay_raw: u16,
    /// stop-display delay (same unit). Taken from the `SP_DCSQ_STM` of
    /// the DCSQ carrying the `STP_DSP` (0x02) command.
    pub stop_delay_raw: u16,
    /// `true` when the control sequence carried at least one
    /// `FSTA_DSP` (Forced Start Display, command `0x00`). A forced
    /// subtitle is one a player should display even when the user has
    /// subtitles disabled — typically used for translations of
    /// on-screen signs / foreign-language dialogue inside an otherwise
    /// untranslated soundtrack. The flag is independent of
    /// `STA_DSP` / `STP_DSP` (an SPU may carry both forced and
    /// non-forced start events at different times); it captures
    /// presence only.
    pub forced_display: bool,
    /// RLE data offsets for top/bottom fields, relative to start of SPU.
    pub top_rle_off: u16,
    pub bot_rle_off: u16,
    /// Set to `true` when the control sequence contained at least one
    /// well-formed CHG_COLCON (0x07) command. The mid-display palette/
    /// alpha mutations the command carries are applied to the rendered
    /// bitmap during decoder canvas paint (see `chg_colcon` below);
    /// this flag also surfaces the fact that an SPU asked for them so
    /// tests / callers can branch on it independently of the structured
    /// data.
    pub saw_chg_colcon: bool,
    /// Parsed CHG_COLCON parameter blocks, in the order they appeared
    /// in the control sequence. Each entry is one vertically-bounded
    /// band (`LN_CTLI`) carrying a list of horizontal start-column
    /// transitions (`PX_CTLI`). Coordinates are in absolute display
    /// (line, column) space, not bitmap-local space — the decoder
    /// intersects them with the SPU's bounding box (`x1..=x2`,
    /// `y1..=y2`) when applying the mutations to the rendered canvas.
    pub chg_colcon: Vec<ChgColConBand>,
}

/// One vertically-bounded band in a CHG_COLCON parameter block — a
/// `LN_CTLI` plus the `PX_CTLI` entries that follow it.
#[derive(Clone, Debug, Default)]
pub struct ChgColConBand {
    /// Top-most display line covered by this band (inclusive).
    pub csln: u16,
    /// Bottom-most display line covered by this band (inclusive).
    pub ctln: u16,
    /// Horizontal start-column transitions inside the band, left-to-right.
    pub entries: Vec<ChgColConEntry>,
}

/// One `PX_CTLI`: from `start_col` rightwards (up to the next entry or
/// the right edge of the display area), pixels of each 2-bit RLE value
/// pick up the replacement palette index from `palette_sel` and the
/// replacement alpha nibble from `alpha`, in lieu of the SPU's base
/// SET_COLOR / SET_CONTR selections.
#[derive(Clone, Debug, Default)]
pub struct ChgColConEntry {
    /// Display column where the replacement starts (inclusive).
    pub start_col: u16,
    /// Replacement palette indices into the 16-entry `.idx` palette,
    /// indexed by the 2-bit RLE pixel value (bg / pattern / emp1 / emp2).
    pub palette_sel: [u8; 4],
    /// Replacement alpha nibbles (0..15), indexed by the 2-bit RLE
    /// pixel value, in the same order as `palette_sel`.
    pub alpha: [u8; 4],
}

/// Parse a SPU (one DVD subtitle unit), producing its control state and
/// a decoded width×height indexed bitmap (indices 0..3, mapping through
/// `palette_sel` → the 16-entry idx palette).
pub fn parse_and_decode_spu(spu: &[u8]) -> Result<(Spu, Vec<u8>, (u16, u16))> {
    if spu.len() < 4 {
        return Err(Error::invalid("vobsub SPU: too short"));
    }
    let spu_len = u16::from_be_bytes([spu[0], spu[1]]) as usize;
    let ctrl_off = u16::from_be_bytes([spu[2], spu[3]]) as usize;
    if spu_len > spu.len() || ctrl_off > spu_len || ctrl_off < 4 {
        return Err(Error::invalid("vobsub SPU: inconsistent sizes"));
    }

    let mut out = Spu::default();
    let mut pos = ctrl_off;
    // Track whether `start_delay_raw` has been latched. The spec lets
    // either `FSTA_DSP` (forced) or `STA_DSP` (regular) trigger
    // on-display; whichever appears first inside the SPU's control
    // sequence is the one whose DCSQ delay we keep. A later DCSQ that
    // *also* asserts start-display is treated as a redundant retrigger
    // and ignored. STP_DSP overwrites unconditionally — the latest
    // stop-display in DCSQ traversal order is the actual end time.
    let mut start_latched = false;
    loop {
        if pos + 4 > spu_len {
            break;
        }
        let delay = u16::from_be_bytes([spu[pos], spu[pos + 1]]);
        let next = u16::from_be_bytes([spu[pos + 2], spu[pos + 3]]) as usize;
        let mut cmd_pos = pos + 4;
        while cmd_pos < spu_len {
            let cmd = spu[cmd_pos];
            cmd_pos += 1;
            match cmd {
                0x00 => {
                    // FSTA_DSP — Forced Start Display. The SPU asks the
                    // player to show this cue even when subtitles are
                    // otherwise disabled. The flag captures presence
                    // anywhere in the control sequence; the timing of
                    // the forced display is the DCSQ's STM, which we
                    // latch into `start_delay_raw` on the same rules
                    // that govern STA_DSP.
                    out.forced_display = true;
                    if !start_latched {
                        out.start_delay_raw = delay;
                        start_latched = true;
                    }
                }
                0x01 => {
                    // STA_DSP — Start Display. The DCSQ's STM is the
                    // delay (in 1024/90000 s units from the SPU's PTS)
                    // before the bitmap appears on screen. We latch the
                    // *first* start-display encountered in DCSQ
                    // traversal order; subsequent start-display
                    // commands inside the same SPU are tolerated but
                    // do not overwrite the latched value.
                    if !start_latched {
                        out.start_delay_raw = delay;
                        start_latched = true;
                    }
                }
                0x02 => {
                    // STP_DSP — Stop Display. The latest stop in DCSQ
                    // traversal order wins; an authoring tool that
                    // emits multiple stops (e.g. when revising the
                    // end time inside one SPU) intends the last one
                    // to take effect.
                    out.stop_delay_raw = delay;
                }
                0x03 => {
                    if cmd_pos + 2 > spu_len {
                        return Err(Error::invalid("vobsub SPU: palette command truncated"));
                    }
                    let b0 = spu[cmd_pos];
                    let b1 = spu[cmd_pos + 1];
                    cmd_pos += 2;
                    out.palette_sel[0] = b0 >> 4; // bg
                    out.palette_sel[1] = b0 & 0x0F; // pattern
                    out.palette_sel[2] = b1 >> 4; // emp1
                    out.palette_sel[3] = b1 & 0x0F; // emp2
                }
                0x04 => {
                    if cmd_pos + 2 > spu_len {
                        return Err(Error::invalid("vobsub SPU: alpha command truncated"));
                    }
                    let b0 = spu[cmd_pos];
                    let b1 = spu[cmd_pos + 1];
                    cmd_pos += 2;
                    out.alpha[0] = b0 >> 4;
                    out.alpha[1] = b0 & 0x0F;
                    out.alpha[2] = b1 >> 4;
                    out.alpha[3] = b1 & 0x0F;
                }
                0x05 => {
                    if cmd_pos + 6 > spu_len {
                        return Err(Error::invalid("vobsub SPU: coords command truncated"));
                    }
                    let b0 = spu[cmd_pos];
                    let b1 = spu[cmd_pos + 1];
                    let b2 = spu[cmd_pos + 2];
                    let b3 = spu[cmd_pos + 3];
                    let b4 = spu[cmd_pos + 4];
                    let b5 = spu[cmd_pos + 5];
                    cmd_pos += 6;
                    out.x1 = (((b0 as u16) << 4) | ((b1 as u16) >> 4)) & 0x0FFF;
                    out.x2 = ((((b1 as u16) & 0x0F) << 8) | (b2 as u16)) & 0x0FFF;
                    out.y1 = (((b3 as u16) << 4) | ((b4 as u16) >> 4)) & 0x0FFF;
                    out.y2 = ((((b4 as u16) & 0x0F) << 8) | (b5 as u16)) & 0x0FFF;
                }
                0x06 => {
                    if cmd_pos + 4 > spu_len {
                        return Err(Error::invalid("vobsub SPU: rle-offsets command truncated"));
                    }
                    out.top_rle_off = u16::from_be_bytes([spu[cmd_pos], spu[cmd_pos + 1]]);
                    out.bot_rle_off = u16::from_be_bytes([spu[cmd_pos + 2], spu[cmd_pos + 3]]);
                    cmd_pos += 4;
                }
                0x07 => {
                    // CHG_COLCON — change colour/contrast within
                    // rectangular sub-areas of the display. The command
                    // is self-delimiting: a 2-byte total parameter size
                    // (which includes the two size bytes themselves)
                    // follows the command byte, then `size - 2` bytes of
                    // LN_CTLI / PX_CTLI parameters terminated by the
                    // LN_CTLI sentinel `0F FF FF FF`. We parse the
                    // payload into structured bands here and apply the
                    // palette/alpha replacements during canvas paint;
                    // the original byte-count is still respected so the
                    // rest of the control sequence stays in lock-step
                    // even when the parser tolerates an unterminated
                    // payload (some streams pad the payload area without
                    // emitting the explicit sentinel).
                    if cmd_pos + 2 > spu_len {
                        return Err(Error::invalid("vobsub SPU: CHG_COLCON size word truncated"));
                    }
                    let size = u16::from_be_bytes([spu[cmd_pos], spu[cmd_pos + 1]]) as usize;
                    // The size includes the two size bytes themselves; a
                    // value < 2 would underflow when computing the
                    // payload length, and a value of exactly 2 means
                    // zero parameter bytes (still a valid edge case to
                    // tolerate).
                    if size < 2 {
                        return Err(Error::invalid("vobsub SPU: CHG_COLCON size word < 2"));
                    }
                    let new_pos = cmd_pos.saturating_add(size);
                    if new_pos > spu_len {
                        return Err(Error::invalid(
                            "vobsub SPU: CHG_COLCON parameters truncated",
                        ));
                    }
                    let payload = &spu[cmd_pos + 2..new_pos];
                    let bands = parse_chg_colcon_payload(payload)?;
                    out.saw_chg_colcon = true;
                    out.chg_colcon.extend(bands);
                    cmd_pos = new_pos;
                }
                0xFF => {
                    break;
                }
                _ => {
                    // Unknown command — bail to avoid desync.
                    return Err(Error::invalid(format!(
                        "vobsub SPU: unknown command 0x{:02X}",
                        cmd
                    )));
                }
            }
        }
        // The last DCSQ in the chain has `SP_NXT_DCSQ_SA` pointing at
        // itself (per the SPU spec). Bail out on any non-advancing
        // pointer to avoid spinning on a self-referential terminator.
        if next <= pos {
            break;
        }
        pos = next;
    }

    // Decode the bitmap.
    if out.x2 < out.x1 || out.y2 < out.y1 {
        return Err(Error::invalid("vobsub SPU: inverted coords"));
    }
    let width = (out.x2 - out.x1 + 1) as usize;
    let height = (out.y2 - out.y1 + 1) as usize;
    let mut pixels = vec![0u8; width * height];
    if width > 0 && height > 0 {
        // The two field-data pointers come straight off the wire
        // (`SET_DSPXA`, §"06 - SET_DSPXA"), so neither is trustworthy: a
        // corrupt or malicious SPU can point either past the unit or before
        // the control table. Both pixel fields live below the control table
        // (`ctrl_off`), so clamp every slice bound to `[0, ctrl_off]` and
        // guard against an inverted range before indexing. `ctrl_off` is
        // already `<= spu.len()` (validated at the top), so the clamped
        // bounds are always in-range for `spu`.
        let top_off = out.top_rle_off as usize;
        let bot_off = out.bot_rle_off as usize;
        if top_off >= spu_len {
            return Err(Error::invalid("vobsub SPU: top offset out of range"));
        }
        // The top field runs from its pointer to the bottom-field pointer
        // (when that sits after it) or to the control table, whichever ends
        // the field data first.
        let raw_top_end = if bot_off > top_off { bot_off } else { ctrl_off };
        let top_start = top_off.min(ctrl_off);
        let top_end = raw_top_end.min(ctrl_off).max(top_start);
        let top_bytes = &spu[top_start..top_end];
        let bot_bytes = if bot_off > 0 && bot_off < ctrl_off {
            &spu[bot_off..ctrl_off]
        } else {
            &[][..]
        };
        decode_rle_field(top_bytes, width, height, 0, 2, &mut pixels)?;
        if !bot_bytes.is_empty() {
            decode_rle_field(bot_bytes, width, height, 1, 2, &mut pixels)?;
        }
    }
    Ok((out, pixels, (width as u16, height as u16)))
}

/// Parse a CHG_COLCON parameter payload (the bytes *after* the 2-byte
/// total-size word, i.e. `size - 2` bytes) into a list of bands.
///
/// Each band starts with one 4-byte LN_CTLI header in the form
/// `0sss ntt t` (4-bit nibbles, big-endian), where:
///
/// * `sss` (12 bits) is `csln`, the inclusive top-most display line;
/// * `n` (4 bits) is the number of `PX_CTLI` entries that follow;
/// * `ttt` (12 bits) is `ctln`, the inclusive bottom-most display line.
///
/// Each `PX_CTLI` is 6 bytes: a 2-byte big-endian start column, a 2-byte
/// SET_COLOR-style nibble pair (`bg|pat`, `emp1|emp2`), and a 2-byte
/// SET_CONTR-style alpha nibble pair (same byte order). The sentinel
/// `0F FF FF FF` LN_CTLI terminates the list.
///
/// We tolerate two well-formed payload-end shapes: the explicit
/// sentinel, and a clean exhaustion of the payload bytes (some
/// authoring tools pad the payload area to a fixed size and rely on the
/// outer 2-byte size word to bound it). Any other malformation —
/// truncated LN_CTLI or PX_CTLI, `ctln < csln`, or a non-decreasing
/// start column inside a band — is reported as `Error::invalid`.
fn parse_chg_colcon_payload(payload: &[u8]) -> Result<Vec<ChgColConBand>> {
    let mut bands = Vec::new();
    let mut pos = 0usize;
    while pos < payload.len() {
        // Need at least one LN_CTLI (4 bytes).
        if pos + 4 > payload.len() {
            return Err(Error::invalid(
                "vobsub SPU: CHG_COLCON LN_CTLI header truncated",
            ));
        }
        let h0 = payload[pos];
        let h1 = payload[pos + 1];
        let h2 = payload[pos + 2];
        let h3 = payload[pos + 3];
        // Sentinel `0F FF FF FF` terminates the list.
        if h0 == 0x0F && h1 == 0xFF && h2 == 0xFF && h3 == 0xFF {
            // Sentinel consumed; any bytes after it are payload padding.
            break;
        }
        // Top 4 bits MUST be zero per the spec layout; if they aren't
        // the byte stream is desynchronised and we'd otherwise produce
        // junk csln values.
        if (h0 & 0xF0) != 0 {
            return Err(Error::invalid(
                "vobsub SPU: CHG_COLCON LN_CTLI high nibble must be zero",
            ));
        }
        let csln = (((h0 as u16) << 8) | (h1 as u16)) & 0x0FFF;
        let n = (h2 >> 4) & 0x0F;
        let ctln = ((((h2 as u16) & 0x0F) << 8) | (h3 as u16)) & 0x0FFF;
        if ctln < csln {
            return Err(Error::invalid("vobsub SPU: CHG_COLCON LN_CTLI ctln < csln"));
        }
        pos += 4;
        let mut entries = Vec::with_capacity(n as usize);
        let mut last_start: Option<u16> = None;
        for _ in 0..n {
            if pos + 6 > payload.len() {
                return Err(Error::invalid("vobsub SPU: CHG_COLCON PX_CTLI truncated"));
            }
            let start_col = u16::from_be_bytes([payload[pos], payload[pos + 1]]);
            let c0 = payload[pos + 2];
            let c1 = payload[pos + 3];
            let a0 = payload[pos + 4];
            let a1 = payload[pos + 5];
            pos += 6;
            // PX_CTLI entries inside a band run left-to-right; the next
            // entry's start_col must be strictly greater than the one
            // before it. Otherwise a later entry would silently never
            // get applied (its column range would be empty or inverted).
            if let Some(prev) = last_start {
                if start_col <= prev {
                    return Err(Error::invalid(
                        "vobsub SPU: CHG_COLCON PX_CTLI start_col not strictly increasing",
                    ));
                }
            }
            last_start = Some(start_col);
            entries.push(ChgColConEntry {
                start_col,
                // SET_COLOR-byte convention (matches the 0x03 handler):
                // byte 0 = bg|pattern (high|low nibble),
                // byte 1 = emp1|emp2.
                palette_sel: [c0 >> 4, c0 & 0x0F, c1 >> 4, c1 & 0x0F],
                // SET_CONTR-byte convention (matches the 0x04 handler).
                alpha: [a0 >> 4, a0 & 0x0F, a1 >> 4, a1 & 0x0F],
            });
        }
        bands.push(ChgColConBand {
            csln,
            ctln,
            entries,
        });
    }
    Ok(bands)
}

/// Resolve the active replacement palette / alpha for one
/// canvas-bitmap pixel given the SPU's CHG_COLCON bands. Returns
/// `Some((palette_sel, alpha))` if any band covers `(disp_x, disp_y)`,
/// else `None` (meaning: fall back to the base SET_COLOR / SET_CONTR).
///
/// `disp_x` / `disp_y` are absolute display coordinates (i.e.
/// `bitmap_x + spu.x1`, `bitmap_y + spu.y1`).
///
/// When multiple bands match the same line, the last one wins — this is
/// the natural read-order behaviour for a serially-applied list of
/// replacements. Inside a band, the active entry is the right-most one
/// whose `start_col` is `<= disp_x`.
fn chg_colcon_lookup(
    bands: &[ChgColConBand],
    disp_x: u16,
    disp_y: u16,
) -> Option<([u8; 4], [u8; 4])> {
    let mut hit: Option<([u8; 4], [u8; 4])> = None;
    for band in bands {
        if disp_y < band.csln || disp_y > band.ctln {
            continue;
        }
        // Find the right-most entry whose start_col is <= disp_x.
        let mut chosen: Option<&ChgColConEntry> = None;
        for entry in &band.entries {
            if entry.start_col <= disp_x {
                chosen = Some(entry);
            } else {
                break;
            }
        }
        if let Some(entry) = chosen {
            hit = Some((entry.palette_sel, entry.alpha));
        }
    }
    hit
}

/// Decode a VobSub RLE field into the `pixels` buffer. The VobSub RLE
/// encodes pairs of (count, colour) where colour is 2 bits and count
/// uses 2, 4, 6, or 14 bits depending on prefix. A `count == 0` run
/// means "fill to end of line".
fn decode_rle_field(
    buf: &[u8],
    width: usize,
    height: usize,
    start_row: usize,
    row_step: usize,
    pixels: &mut [u8],
) -> Result<()> {
    let mut bits = NibbleReader::new(buf);
    let mut row = start_row;
    let mut col = 0usize;
    while row < height {
        let first = bits.read(4)?;
        let (count, colour) = if first >= 4 {
            (first >> 2, (first & 0x03) as u8)
        } else if first > 0 {
            // One more nibble for count.
            let n1 = bits.read(4)?;
            let combined = (first << 4) | n1;
            (combined >> 2, (combined & 0x03) as u8)
        } else {
            let n1 = bits.read(4)?;
            if n1 >= 4 {
                let combined = (first << 8) | (n1 << 4) | bits.read(4)?;
                (combined >> 2, (combined & 0x03) as u8)
            } else if n1 > 0 {
                let combined = (first << 12) | (n1 << 8) | (bits.read(4)? << 4) | bits.read(4)?;
                (combined >> 2, (combined & 0x03) as u8)
            } else {
                // rest-of-line: fill to width - col.
                let n2 = bits.read(4)?;
                let combined = (first << 12) | (n1 << 8) | (n2 << 4) | bits.read(4)?;
                (0, (combined & 0x03) as u8)
            }
        };
        let run = if count == 0 {
            width.saturating_sub(col)
        } else {
            count as usize
        };
        let end = (col + run).min(width);
        if row < height {
            let base = row * width + col;
            for px in &mut pixels[base..base + (end - col)] {
                *px = colour;
            }
        }
        col = end;
        if col >= width {
            // Align to next byte at end-of-line.
            bits.align();
            col = 0;
            row += row_step;
            if run == 0 {
                // fill-to-end was explicit; continue to next row.
            }
        }
    }
    Ok(())
}

struct NibbleReader<'a> {
    buf: &'a [u8],
    // half-byte cursor
    pos: usize,
}

impl<'a> NibbleReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn read(&mut self, n: u32) -> Result<u32> {
        debug_assert!(n == 4);
        if self.pos / 2 >= self.buf.len() {
            return Err(Error::invalid("vobsub: RLE bitstream ran out"));
        }
        let b = self.buf[self.pos / 2];
        let nibble = if self.pos % 2 == 0 { b >> 4 } else { b & 0x0F };
        self.pos += 1;
        Ok(nibble as u32)
    }

    fn align(&mut self) {
        if self.pos % 2 != 0 {
            self.pos += 1;
        }
    }
}

// --- container (.idx + .sub demuxer) -----------------------------------

/// Register the VobSub demuxer + extension mappings.
pub fn register_container(reg: &mut ContainerRegistry) {
    reg.register_demuxer("vobsub", open_vobsub);
    reg.register_extension("idx", "vobsub");
    reg.register_extension("sub", "vobsub");
    reg.register_probe("vobsub", probe_vobsub);
}

fn probe_vobsub(p: &ProbeData) -> ProbeScore {
    // .idx files start with "# VobSub index file" on idxsub's output or
    // the line "size:" early in the file. Combined with the extension
    // we score confidently.
    let s = std::str::from_utf8(p.buf).ok().unwrap_or("");
    let hit = s.contains("# VobSub index file")
        || s.contains("\nsize:")
        || s.starts_with("size:")
        || s.contains("\ntimestamp:");
    match (hit, p.ext) {
        (true, Some("idx")) => 100,
        (true, _) => 75,
        (false, Some("idx")) => 25,
        _ => 0,
    }
}

fn open_vobsub(
    mut input: Box<dyn ReadSeek>,
    _codecs: &dyn CodecResolver,
) -> Result<Box<dyn Demuxer>> {
    // We assume the input is the `.idx` file (text). We read it and then
    // look for the matching `.sub` alongside by filename. Since the
    // `ReadSeek` trait doesn't expose a path, the only source we can
    // consult is the content itself — if no companion file is resolvable
    // we fall back to constructing an empty SPU stream so the caller
    // gets at least the stream info and palette back.
    input.seek(SeekFrom::Start(0))?;
    let mut buf = Vec::new();
    input.read_to_end(&mut buf)?;
    let text = String::from_utf8_lossy(&buf).into_owned();
    let idx = parse_idx(&text)?;

    // Try to find the `.sub` alongside. We inspect the idx body for a
    // leading `# path: ...` comment some tools emit, and otherwise
    // produce a packet list from any embedded `# sub-data: <hex>` lines
    // (our own test fixtures use that convention). If none, the stream
    // carries no packets.
    let sub_path = find_sub_alongside(&text);
    let sub_bytes = match sub_path {
        Some(p) => std::fs::read(&p).ok().unwrap_or_default(),
        None => extract_inline_sub(&text).unwrap_or_default(),
    };

    let packets = build_packets(&idx, &sub_bytes);

    let (w, h) = idx.size;
    let mut params = CodecParameters::video(CodecId::new(VOBSUB_CODEC_ID));
    params.media_type = MediaType::Subtitle;
    params.width = Some(w as u32);
    params.height = Some(h as u32);
    params.pixel_format = Some(PixelFormat::Rgba);
    // Pack the 16-entry RGB palette into extradata so the decoder can
    // be built from stream info alone.
    let mut extra = Vec::with_capacity(48);
    for entry in &idx.palette_rgb {
        extra.extend_from_slice(entry);
    }
    params.extradata = extra;

    let total_us = packets.back().and_then(|p| p.pts).unwrap_or(0);
    let stream = StreamInfo {
        index: 0,
        time_base: TimeBase::new(1, 1_000_000),
        duration: Some(total_us),
        start_time: Some(0),
        params,
    };

    Ok(Box::new(VobSubDemuxer {
        streams: [stream],
        packets,
    }))
}

fn find_sub_alongside(idx_text: &str) -> Option<PathBuf> {
    // Look for our test convention: `# idx-path: /some/path.idx`.
    for line in idx_text.lines() {
        if let Some(path) = line.strip_prefix("# idx-path:") {
            let base = Path::new(path.trim()).with_extension("sub");
            return Some(base);
        }
    }
    None
}

fn extract_inline_sub(idx_text: &str) -> Option<Vec<u8>> {
    // `# sub-hex: <hex bytes>` — ours-only convention for in-tests.
    for line in idx_text.lines() {
        if let Some(rest) = line.strip_prefix("# sub-hex:") {
            return decode_hex(rest.trim());
        }
    }
    None
}

fn decode_hex(s: &str) -> Option<Vec<u8>> {
    let clean: String = s.chars().filter(|c| !c.is_whitespace()).collect();
    if clean.len() % 2 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(clean.len() / 2);
    for chunk in clean.as_bytes().chunks(2) {
        let s = std::str::from_utf8(chunk).ok()?;
        out.push(u8::from_str_radix(s, 16).ok()?);
    }
    Some(out)
}

fn build_packets(idx: &VobSubIdx, sub: &[u8]) -> VecDeque<Packet> {
    let tb = TimeBase::new(1, 1_000_000);
    let mut packets = VecDeque::new();
    for (i, (start_us, filepos)) in idx.cues.iter().enumerate() {
        let fp = *filepos as usize;
        if fp >= sub.len() {
            continue;
        }
        // Extract the SPU payload: drop the MPEG-PS pack/PES framing if
        // present (we accept either "raw SPU" or a minimal PES-wrapped
        // form). For the raw form the cue filepos points directly at
        // the SPU size u16.
        let spu = extract_spu(&sub[fp..]).unwrap_or_else(|| sub[fp..].to_vec());
        let mut pkt = Packet::new(0, tb, spu);
        pkt.pts = Some(*start_us);
        pkt.dts = Some(*start_us);
        pkt.flags.keyframe = true;
        if i + 1 < idx.cues.len() {
            let next = idx.cues[i + 1].0;
            pkt.duration = Some((next - *start_us).max(0));
        }
        packets.push_back(pkt);
    }
    packets
}

/// Extract an SPU blob from a buffer that may be either a raw SPU or a
/// MPEG-PS pack + PES private_stream_1 wrapper.
fn extract_spu(buf: &[u8]) -> Option<Vec<u8>> {
    // Raw SPU form: first 2 bytes = SPU length, and SPU length <= buf.len().
    if buf.len() >= 4 {
        let spu_len = u16::from_be_bytes([buf[0], buf[1]]) as usize;
        if spu_len >= 4 && spu_len <= buf.len() {
            return Some(buf[..spu_len].to_vec());
        }
    }
    // MPEG-PS pack + PES private_stream_1 form: walk the PS packets and
    // concatenate the PES payloads (dropping the 1-byte substream-id
    // prefix DVD uses inside private_stream_1) until we have enough to
    // hold a full SPU (its first 2 bytes give the total length).
    extract_spu_from_ps(buf)
}

/// Walk an MPEG-PS stream, concatenating DVD sub payloads from every
/// private_stream_1 PES packet until one full SPU has been recovered.
fn extract_spu_from_ps(buf: &[u8]) -> Option<Vec<u8>> {
    let mut cur = 0usize;
    let mut spu: Vec<u8> = Vec::new();
    let mut target: Option<usize> = None;

    while cur + 4 <= buf.len() {
        // All PS start codes share the 0x000001 prefix.
        if buf[cur] != 0 || buf[cur + 1] != 0 || buf[cur + 2] != 1 {
            return None;
        }
        let code = buf[cur + 3];
        cur += 4;
        match code {
            // Pack header (0xBA). MPEG-2 form: 10xx then 9 bytes, then a
            // 3-bit stuffing-length followed by that many stuffing bytes.
            // MPEG-1 form: starts with 0010xxxx and is 8 bytes total.
            0xBA => {
                if cur >= buf.len() {
                    return None;
                }
                if (buf[cur] & 0xC0) == 0x40 {
                    // MPEG-2: 9 fixed bytes + stuffing (low 3 bits of byte 9).
                    if cur + 10 > buf.len() {
                        return None;
                    }
                    let stuffing = (buf[cur + 9] & 0x07) as usize;
                    cur += 10 + stuffing;
                } else if (buf[cur] & 0xF0) == 0x20 {
                    // MPEG-1: 8 fixed bytes.
                    cur += 8;
                } else {
                    return None;
                }
            }
            // System header — 2-byte length followed by that many bytes.
            0xBB => {
                if cur + 2 > buf.len() {
                    return None;
                }
                let len = u16::from_be_bytes([buf[cur], buf[cur + 1]]) as usize;
                cur += 2 + len;
            }
            // Private stream 1 — DVD subs. Length + PES header + 1-byte
            // substream id, then the SPU payload fragment.
            0xBD => {
                if cur + 2 > buf.len() {
                    return None;
                }
                let pes_len = u16::from_be_bytes([buf[cur], buf[cur + 1]]) as usize;
                cur += 2;
                let pes_end = cur + pes_len;
                if pes_end > buf.len() {
                    return None;
                }
                let body = parse_pes_body(&buf[cur..pes_end])?;
                // DVD sub packets carry a 1-byte substream id at body[0];
                // valid ids are 0x20..0x3F.
                if body.is_empty() {
                    cur = pes_end;
                    continue;
                }
                let substream = body[0];
                if !(0x20..=0x3F).contains(&substream) {
                    cur = pes_end;
                    continue;
                }
                spu.extend_from_slice(&body[1..]);
                if target.is_none() && spu.len() >= 2 {
                    target = Some(u16::from_be_bytes([spu[0], spu[1]]) as usize);
                }
                if let Some(t) = target {
                    if spu.len() >= t {
                        spu.truncate(t);
                        return Some(spu);
                    }
                }
                cur = pes_end;
            }
            // Padding / other PES packets — just skip by length.
            0xBE | 0xBF => {
                if cur + 2 > buf.len() {
                    return None;
                }
                let len = u16::from_be_bytes([buf[cur], buf[cur + 1]]) as usize;
                cur += 2 + len;
            }
            // MPEG program end.
            0xB9 => break,
            _ => {
                if cur + 2 > buf.len() {
                    return None;
                }
                let len = u16::from_be_bytes([buf[cur], buf[cur + 1]]) as usize;
                cur += 2 + len;
            }
        }
    }
    None
}

/// Strip the PES header of an MPEG-1 or MPEG-2 PES packet body, returning
/// a slice starting at the payload byte (the DVD substream-id byte for
/// private_stream_1).
fn parse_pes_body(pes: &[u8]) -> Option<&[u8]> {
    if pes.is_empty() {
        return None;
    }
    // MPEG-2 PES: starts with 0b10xxxxxx. Header format:
    //   flags1 (1) | flags2 (1) | hdr_data_len (1) | hdr_data (...).
    if (pes[0] & 0xC0) == 0x80 {
        if pes.len() < 3 {
            return None;
        }
        let hdr_len = pes[2] as usize;
        let start = 3 + hdr_len;
        if start > pes.len() {
            return None;
        }
        return Some(&pes[start..]);
    }
    // MPEG-1 PES: leading stuffing (0xFF) up to 16 bytes, then optional
    // STD buffer size (2 bytes, first byte & 0xC0 == 0x40), then
    // optional PTS/DTS (flags 0x20/0x30/0x10 leading bits).
    let mut i = 0usize;
    while i < pes.len() && i < 16 && pes[i] == 0xFF {
        i += 1;
    }
    if i >= pes.len() {
        return None;
    }
    if (pes[i] & 0xC0) == 0x40 {
        i += 2;
    }
    if i >= pes.len() {
        return None;
    }
    let b = pes[i];
    if (b & 0xF0) == 0x20 {
        i += 5;
    } else if (b & 0xF0) == 0x30 {
        i += 10;
    } else if b == 0x0F {
        i += 1;
    }
    if i > pes.len() {
        return None;
    }
    Some(&pes[i..])
}

struct VobSubDemuxer {
    streams: [StreamInfo; 1],
    packets: VecDeque<Packet>,
}

impl Demuxer for VobSubDemuxer {
    fn format_name(&self) -> &str {
        "vobsub"
    }

    fn streams(&self) -> &[StreamInfo] {
        &self.streams
    }

    fn next_packet(&mut self) -> Result<Packet> {
        self.packets.pop_front().ok_or(Error::Eof)
    }

    fn duration_micros(&self) -> Option<i64> {
        self.streams[0].duration
    }
}

// --- decoder -----------------------------------------------------------

/// Build a VobSub decoder.
pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    let mut palette = [[0u8; 3]; 16];
    if params.extradata.len() >= 48 {
        for (i, p) in palette.iter_mut().enumerate() {
            *p = [
                params.extradata[i * 3],
                params.extradata[i * 3 + 1],
                params.extradata[i * 3 + 2],
            ];
        }
    } else {
        // Fallback grayscale ramp so tests without a real idx still decode.
        for (i, p) in palette.iter_mut().enumerate() {
            let g = (i * 17) as u8;
            *p = [g, g, g];
        }
    }
    Ok(Box::new(VobSubDecoder {
        codec_id: CodecId::new(VOBSUB_CODEC_ID),
        palette,
        pending: VecDeque::new(),
        eof: false,
    }))
}

struct VobSubDecoder {
    codec_id: CodecId,
    palette: [[u8; 3]; 16],
    pending: VecDeque<Frame>,
    eof: bool,
}

impl Decoder for VobSubDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        let (spu, pixels, (w, h)) = parse_and_decode_spu(&packet.data)?;
        let mut canvas = vec![0u8; (w as usize) * (h as usize) * 4];
        let bbox_x = spu.x1;
        let bbox_y = spu.y1;
        let has_chg_colcon = !spu.chg_colcon.is_empty();
        for (i, &idx_4) in pixels.iter().enumerate() {
            let which = idx_4 as usize & 0x03;
            // Resolve replacement palette/alpha from CHG_COLCON when the
            // pixel's display-space coordinate falls inside a band.
            // Bitmap-local (x, y) maps to display-space via the SPU's
            // (x1, y1) origin.
            let (pal_arr, alpha_arr) = if has_chg_colcon {
                let bx = (i % (w as usize)) as u16;
                let by = (i / (w as usize)) as u16;
                let disp_x = bbox_x.saturating_add(bx);
                let disp_y = bbox_y.saturating_add(by);
                match chg_colcon_lookup(&spu.chg_colcon, disp_x, disp_y) {
                    Some((p, a)) => (p, a),
                    None => (spu.palette_sel, spu.alpha),
                }
            } else {
                (spu.palette_sel, spu.alpha)
            };
            let pal_idx = pal_arr[which] as usize & 0x0F;
            let alpha4 = alpha_arr[which] & 0x0F;
            let alpha = alpha4 * 17; // 0..15 → 0..255
            if alpha == 0 {
                continue;
            }
            let rgb = self.palette[pal_idx];
            let dst = i * 4;
            canvas[dst] = rgb[0];
            canvas[dst + 1] = rgb[1];
            canvas[dst + 2] = rgb[2];
            canvas[dst + 3] = alpha;
        }
        let frame = VideoFrame {
            pts: packet.pts,
            planes: vec![VideoPlane {
                stride: (w as usize) * 4,
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
        // Each VobSub SPU is self-contained. Drop the ready-frame queue
        // and clear the eof latch; the palette is stream-level config.
        self.pending.clear();
        self.eof = false;
        Ok(())
    }
}

// --- test helpers ------------------------------------------------------

/// Build a tiny SPU with the given width/height and per-pixel palette
/// indices (0..3). Each row has a run of `(w, colour)` ended by a
/// rest-of-line marker — callers whose `w` doesn't fit the 14-bit
/// representation should switch to separate runs.
#[doc(hidden)]
pub fn build_demo_spu(width: u16, height: u16, indices: &[u8]) -> Vec<u8> {
    assert_eq!(indices.len(), (width as usize) * (height as usize));
    fn push_rle_rows(
        out: &mut Vec<u8>,
        indices: &[u8],
        width: u16,
        _height: u16,
        field_rows: impl Iterator<Item = usize>,
    ) {
        let mut bits = NibbleWriter::new();
        for row in field_rows {
            let mut col = 0usize;
            while col < width as usize {
                // Find next run of identical indices.
                let colour = indices[row * width as usize + col];
                let mut run = 1usize;
                while col + run < width as usize
                    && indices[row * width as usize + col + run] == colour
                    && run < 0x3FFF
                {
                    run += 1;
                }
                // Prefer rest-of-line when possible.
                let rest = col + run == width as usize;
                emit_rle(&mut bits, if rest { 0 } else { run as u32 }, colour);
                col += run;
            }
            bits.align();
        }
        bits.finish(out);
    }

    fn emit_rle(w: &mut NibbleWriter, count: u32, colour: u8) {
        let c = (colour & 0x03) as u32;
        // If count is 0 → rest-of-line: encode with 4 leading zeros
        // nibble then 2 bits of colour packed into a nibble.
        if count == 0 {
            // 0000 0000 0000 CC -> 4 nibbles = 0 0 0 c
            w.write(4, 0);
            w.write(4, 0);
            w.write(4, 0);
            w.write(4, c);
            return;
        }
        if count < 4 {
            // 2-bit count + 2-bit colour in one nibble (cc[1:0]Cc[1:0])
            let nib = ((count & 0x3) << 2) | c;
            w.write(4, nib);
            return;
        }
        if count < 16 {
            // 4-bit count + 2-bit colour in two nibbles: 0 count colour
            let val = (count << 2) | c; // 6 bits
            w.write(4, (val >> 4) & 0xF);
            w.write(4, val & 0xF);
            return;
        }
        if count < 64 {
            // 6-bit count + 2-bit colour in two nibbles with leading zero prefix
            let val = (count << 2) | c; // 8 bits
            w.write(4, 0);
            w.write(4, (val >> 4) & 0xF);
            w.write(4, val & 0xF);
            return;
        }
        // 14-bit: count<<2 | c → 16 bits = four nibbles, with leading 0 prefix.
        let val = (count << 2) | c;
        w.write(4, 0);
        w.write(4, (val >> 12) & 0xF);
        w.write(4, (val >> 8) & 0xF);
        w.write(4, (val >> 4) & 0xF);
        w.write(4, val & 0xF);
    }

    struct NibbleWriter {
        nibbles: Vec<u8>,
    }
    impl NibbleWriter {
        fn new() -> Self {
            Self {
                nibbles: Vec::new(),
            }
        }
        fn write(&mut self, _bits: u32, value: u32) {
            self.nibbles.push((value & 0x0F) as u8);
        }
        fn align(&mut self) {
            if self.nibbles.len() % 2 != 0 {
                self.nibbles.push(0);
            }
        }
        fn finish(&self, out: &mut Vec<u8>) {
            for pair in self.nibbles.chunks(2) {
                let hi = pair[0];
                let lo = if pair.len() == 2 { pair[1] } else { 0 };
                out.push((hi << 4) | lo);
            }
        }
    }

    // Build top & bottom RLE byte blocks.
    let mut top_bytes = Vec::new();
    push_rle_rows(
        &mut top_bytes,
        indices,
        width,
        height,
        (0..height as usize).step_by(2),
    );
    let mut bot_bytes = Vec::new();
    push_rle_rows(
        &mut bot_bytes,
        indices,
        width,
        height,
        (1..height as usize).step_by(2),
    );

    // Layout:
    //   [0..2]  SPU length
    //   [2..4]  control offset
    //   [4..]   RLE data (top then bottom)
    //   control: delay=0, next=pos, commands 0x03 palette, 0x04 alpha,
    //     0x05 coords, 0x06 offsets, 0x01 start, 0xFF end.
    let top_off = 4usize;
    let bot_off = top_off + top_bytes.len();
    let ctrl_off = bot_off + bot_bytes.len();
    let mut out = Vec::new();
    out.extend_from_slice(&[0, 0]); // placeholder SPU length
    out.extend_from_slice(&(ctrl_off as u16).to_be_bytes());
    out.extend_from_slice(&top_bytes);
    out.extend_from_slice(&bot_bytes);

    // Single control sequence.
    let ctrl_pos = out.len();
    out.extend_from_slice(&[0, 0]); // delay = 0
                                    // next_offset placeholder — will point back to itself at end.
    out.extend_from_slice(&[0, 0]);
    out.push(0x03); // palette select
    out.push(0x01); // bg=0, pat=1
    out.push(0x32); // emp1=3, emp2=2
    out.push(0x04); // alpha
    out.push(0x00); // bg=0, pat=0xF (full)
    out.push(0xFF);
    // but we actually want pattern+emp alphas to 0xF. rewrite:
    let last = out.len() - 2;
    out[last] = 0x0F; // (bg<<4)|pattern → bg=0 pat=0xF
    out[last + 1] = 0xFF; // emp1=0xF, emp2=0xF
    out.push(0x05); // coords
                    // x1 = 0, x2 = width - 1
    out.push(0);
    out.push((((width - 1) >> 8) as u8) & 0x0F);
    out.push(((width - 1) & 0xFF) as u8);
    // y1 = 0, y2 = height - 1
    out.push(0);
    out.push((((height - 1) >> 8) as u8) & 0x0F);
    out.push(((height - 1) & 0xFF) as u8);
    out.push(0x06); // RLE offsets
    out.extend_from_slice(&(top_off as u16).to_be_bytes());
    out.extend_from_slice(&(bot_off as u16).to_be_bytes());
    out.push(0x01); // start display
    out.push(0xFF);

    // Patch next_offset in control sequence: point at itself to terminate.
    out[ctrl_pos + 2] = (ctrl_pos as u16 >> 8) as u8;
    out[ctrl_pos + 3] = (ctrl_pos as u16 & 0xFF) as u8;
    // Patch SPU length.
    let total = out.len() as u16;
    out[0] = (total >> 8) as u8;
    out[1] = (total & 0xFF) as u8;

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_idx_basics() {
        let text = "\
# VobSub index file
size: 720x480
palette: ff0000, 00ff00, 0000ff, ffffff, 000000, 808080, c0c0c0, 404040, 200020, 800080, a0a0a0, 010203, 040506, 070809, 0a0b0c, 0d0e0f
timestamp: 00:00:01:500, filepos: 000000000
timestamp: 00:00:03:000, filepos: 000000040
";
        let idx = parse_idx(text).unwrap();
        assert_eq!(idx.size, (720, 480));
        assert!(idx.has_palette);
        assert_eq!(idx.palette_rgb[0], [0xff, 0x00, 0x00]);
        assert_eq!(idx.palette_rgb[1], [0x00, 0xff, 0x00]);
        assert_eq!(idx.cues.len(), 2);
        assert_eq!(idx.cues[0].0, 1_500_000);
        assert_eq!(idx.cues[1].0, 3_000_000);
    }

    #[test]
    fn decodes_small_spu() {
        // 2×2 bitmap filled with palette index 1 (pattern colour).
        let indices = [1u8, 1, 1, 1];
        let spu = build_demo_spu(2, 2, &indices);
        let (state, pixels, (w, h)) = parse_and_decode_spu(&spu).unwrap();
        assert_eq!(w, 2);
        assert_eq!(h, 2);
        assert_eq!(pixels, vec![1u8, 1, 1, 1]);
        assert_eq!(state.palette_sel[1], 1);

        let mut params = CodecParameters::video(CodecId::new(VOBSUB_CODEC_ID));
        // 16-colour palette: entry 1 is red.
        let mut extra = vec![0u8; 48];
        extra[3] = 255;
        params.extradata = extra;
        let mut dec = make_decoder(&params).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 1_000_000), spu).with_pts(0);
        dec.send_packet(&pkt).unwrap();
        let frame = dec.receive_frame().unwrap();
        let Frame::Video(v) = frame else {
            panic!("expected video frame");
        };
        assert_eq!(v.planes[0].stride, 2 * 4);
        assert_eq!(v.planes[0].data.len(), 2 * 2 * 4);
        let data = &v.planes[0].data;
        // All pixels red + alpha 255.
        for px in data.chunks(4) {
            assert_eq!(px, &[255, 0, 0, 255], "pixel not red: {:?}", px);
        }
    }

    #[test]
    fn extracts_spu_from_raw() {
        let raw = build_demo_spu(2, 2, &[1u8, 1, 1, 1]);
        let out = extract_spu(&raw).unwrap();
        assert_eq!(out, raw);
    }

    /// The minimised SPU the `vobsub_spu` fuzz target distilled to an
    /// out-of-bounds slice panic: a `SET_DSPXA` whose bottom-field pixel
    /// pointer (`bot_rle_off`) is far past the end of the unit, so
    /// `&spu[bot_off..ctrl_off]` indexed out of range. The field-data
    /// pointers come straight off the wire and are untrusted; the decoder
    /// must clamp them to the control-table offset and never panic.
    #[test]
    fn fuzz_oob_field_pointer_does_not_panic() {
        let spu: &[u8] = &[
            0x00, 0x0f, 0x00, 0x04, 0x00, 0x04, 0x01, 0x00, 0x00, 0x06, 0x00, 0x00, 0x06, 0xff,
            0xff, 0x04, 0xff, 0x00, 0x03, 0x01, 0x00, 0x00, 0x00, 0x7a,
        ];
        // Must return a `Result` (Ok or Err), never panic / index OOB.
        let _ = parse_and_decode_spu(spu);
    }

    /// A `SET_DSPXA` with a wildly out-of-range bottom-field pointer is
    /// tolerated: the bottom field is treated as empty rather than slicing
    /// past the unit. Builds a baseline decodable SPU, then rewrites only
    /// the bottom-field pointer to a huge value and confirms the top field
    /// still decodes without panic.
    #[test]
    fn out_of_range_bottom_pointer_yields_empty_bottom_field() {
        let raw = build_demo_spu(4, 2, &[1u8, 1, 1, 1, 2, 2, 2, 2]);
        // Sanity: the unmodified SPU decodes.
        assert!(parse_and_decode_spu(&raw).is_ok());
        // Find the SET_DSPXA (0x06) command in the control block and corrupt
        // its bottom-field pointer to 0xFFFF. The control table starts at the
        // offset named by bytes [2..4].
        let ctrl_off = u16::from_be_bytes([raw[2], raw[3]]) as usize;
        let mut spu = raw.clone();
        let mut i = ctrl_off + 4; // skip SP_DCSQ_STM + SP_NXT_DCSQ_SA
        while i < spu.len() {
            match spu[i] {
                0x00..=0x02 => i += 1,
                0x03 | 0x04 => i += 3,
                0x05 => i += 7,
                0x06 => {
                    // 06 + top(2) + bot(2); clobber the bottom pointer.
                    spu[i + 3] = 0xff;
                    spu[i + 4] = 0xff;
                    break;
                }
                _ => break,
            }
        }
        // Still no panic, and the top field still produced pixels.
        let res = parse_and_decode_spu(&spu);
        assert!(
            res.is_ok(),
            "corrupt bottom pointer must not error/panic: {res:?}"
        );
    }

    /// Build a tiny SPU that carries one well-formed CHG_COLCON (0x07)
    /// in its control sequence, in addition to the usual palette / alpha
    /// / coords / RLE-offsets / start-display / end run. The CHG_COLCON
    /// parameter block is a single LN_CTLI terminator (`0F FF FF FF`) so
    /// the documented size word is `0x0006` (size word + 4-byte
    /// terminator), giving the decoder a real length to skip.
    ///
    /// Rebuilds the control block by walking the baseline SPU's commands
    /// with the documented per-command argument widths (we can't byte-search
    /// for `0x01` because palette/alpha argument bytes can collide with
    /// command bytes — every nibble is valid argument data).
    fn build_spu_with_chg_colcon() -> Vec<u8> {
        let base = build_demo_spu(2, 2, &[1u8, 1, 1, 1]);
        let base_ctrl = u16::from_be_bytes([base[2], base[3]]) as usize;
        let rle_region = &base[4..base_ctrl];

        // Walk the baseline's commands and find where 0x01 (start-display)
        // sits, taking each command's documented length into account.
        let cmd_region = &base[base_ctrl + 4..];
        let mut walk = 0usize;
        let start_idx;
        loop {
            assert!(walk < cmd_region.len(), "ran off end without START");
            let cmd = cmd_region[walk];
            match cmd {
                0x00 | 0x01 => {
                    start_idx = walk;
                    break;
                }
                0x02 => walk += 1,
                0x03 | 0x04 => walk += 1 + 2,
                0x05 => walk += 1 + 6,
                0x06 => walk += 1 + 4,
                0xFF => panic!("baseline ended before START"),
                other => panic!("unexpected baseline cmd 0x{other:02X}"),
            }
            if cmd == 0x02 || (0x03..=0x06).contains(&cmd) {
                // walk already advanced
            }
        }

        let mut new_cmds = Vec::new();
        new_cmds.extend_from_slice(&cmd_region[..start_idx]);
        // CHG_COLCON: command + 2-byte size (incl. size word) + payload
        new_cmds.push(0x07);
        let chg_params: [u8; 4] = [0x0F, 0xFF, 0xFF, 0xFF];
        let size = (2 + chg_params.len()) as u16;
        new_cmds.extend_from_slice(&size.to_be_bytes());
        new_cmds.extend_from_slice(&chg_params);
        // Append the rest of the baseline cmds (start through 0xFF).
        new_cmds.extend_from_slice(&cmd_region[start_idx..]);
        // Sanity: cmd_region already ends with 0xFF so new_cmds does too.
        assert_eq!(*new_cmds.last().unwrap(), 0xFF);

        let mut out = Vec::new();
        out.extend_from_slice(&[0, 0]); // placeholder length
        let new_ctrl_off = 4 + rle_region.len();
        out.extend_from_slice(&(new_ctrl_off as u16).to_be_bytes());
        out.extend_from_slice(rle_region);
        let ctrl_pos = out.len();
        out.extend_from_slice(&[0, 0]); // delay = 0
        out.extend_from_slice(&[0, 0]); // next placeholder
        out.extend_from_slice(&new_cmds);
        // Patch next-offset to point at this sequence (terminator).
        out[ctrl_pos + 2] = (ctrl_pos as u16 >> 8) as u8;
        out[ctrl_pos + 3] = (ctrl_pos as u16 & 0xFF) as u8;
        let total = out.len() as u16;
        out[0] = (total >> 8) as u8;
        out[1] = (total & 0xFF) as u8;
        out
    }

    #[test]
    fn chg_colcon_command_is_skipped_not_error() {
        let spu = build_spu_with_chg_colcon();
        let (state, pixels, (w, h)) =
            parse_and_decode_spu(&spu).expect("CHG_COLCON-bearing SPU should decode");
        assert_eq!((w, h), (2, 2));
        // The base palette/alpha selection is unchanged by the
        // length-skipped CHG_COLCON, so the bitmap is identical to the
        // CHG_COLCON-free version of the same RLE.
        let baseline = build_demo_spu(2, 2, &[1u8, 1, 1, 1]);
        let (_, baseline_px, _) = parse_and_decode_spu(&baseline).unwrap();
        assert_eq!(pixels, baseline_px);
        assert!(
            state.saw_chg_colcon,
            "Spu.saw_chg_colcon should be true when 0x07 was parsed"
        );
        // Baseline SPU's flag stays false.
        let (baseline_state, _, _) = parse_and_decode_spu(&baseline).unwrap();
        assert!(!baseline_state.saw_chg_colcon);
    }

    #[test]
    fn chg_colcon_truncated_size_word_errors() {
        // Build a control sequence whose final command is 0x07 with
        // only 1 trailing byte before the SPU ends — the 2-byte size
        // word can't be read.
        let mut out = Vec::new();
        out.extend_from_slice(&[0, 0]); // placeholder length
        out.extend_from_slice(&[0, 4]); // control_offset = 4 (no RLE)
                                        // Control block.
        out.extend_from_slice(&[0, 0, 0, 0]); // delay + next placeholder
        out.push(0x07); // CHG_COLCON with no size bytes following
        let total = out.len() as u16;
        out[0] = (total >> 8) as u8;
        out[1] = (total & 0xFF) as u8;
        // Patch next-offset = ctrl_pos = 4.
        out[6] = 0;
        out[7] = 4;
        let err = parse_and_decode_spu(&out).unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("CHG_COLCON size word truncated"),
            "expected truncated-size error, got: {msg}"
        );
    }

    #[test]
    fn chg_colcon_truncated_payload_errors() {
        // CHG_COLCON with size=0x0010 (announces 14 bytes of params)
        // but only 2 trailing bytes available — payload truncated.
        let mut out = Vec::new();
        out.extend_from_slice(&[0, 0]); // placeholder length
        out.extend_from_slice(&[0, 4]); // control_offset
        out.extend_from_slice(&[0, 0, 0, 0]); // delay + next placeholder
        out.push(0x07);
        out.extend_from_slice(&[0x00, 0x10]); // size = 16 bytes (way more than left)
        out.extend_from_slice(&[0xAA, 0xBB]); // 2 trailing bytes
        let total = out.len() as u16;
        out[0] = (total >> 8) as u8;
        out[1] = (total & 0xFF) as u8;
        out[6] = 0;
        out[7] = 4;
        let err = parse_and_decode_spu(&out).unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("CHG_COLCON parameters truncated"),
            "expected payload-truncated error, got: {msg}"
        );
    }

    #[test]
    fn chg_colcon_zero_size_word_errors() {
        // size word of 0 (or 1) would underflow the payload calc.
        let mut out = Vec::new();
        out.extend_from_slice(&[0, 0]);
        out.extend_from_slice(&[0, 4]);
        out.extend_from_slice(&[0, 0, 0, 0]);
        out.push(0x07);
        out.extend_from_slice(&[0x00, 0x00]); // size = 0
        let total = out.len() as u16;
        out[0] = (total >> 8) as u8;
        out[1] = (total & 0xFF) as u8;
        out[6] = 0;
        out[7] = 4;
        let err = parse_and_decode_spu(&out).unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("CHG_COLCON size word < 2"),
            "expected size<2 error, got: {msg}"
        );
    }

    #[test]
    fn chg_colcon_zero_payload_size_two_is_tolerated() {
        // size = 2 means "just the size word, no parameter bytes". The
        // command is still self-delimiting; decode should succeed and
        // saw_chg_colcon should be set. We replay the same controlled
        // rebuild used in `build_spu_with_chg_colcon` but inject a
        // smaller CHG_COLCON, to keep the test independent of the
        // baseline's exact command layout.
        let base = build_demo_spu(2, 2, &[1u8, 1, 1, 1]);
        let base_ctrl = u16::from_be_bytes([base[2], base[3]]) as usize;
        let rle_region = &base[4..base_ctrl];
        let cmd_region = &base[base_ctrl + 4..];
        let mut walk = 0usize;
        let start_idx;
        loop {
            assert!(walk < cmd_region.len());
            let cmd = cmd_region[walk];
            match cmd {
                0x00 | 0x01 => {
                    start_idx = walk;
                    break;
                }
                0x02 => walk += 1,
                0x03 | 0x04 => walk += 1 + 2,
                0x05 => walk += 1 + 6,
                0x06 => walk += 1 + 4,
                _ => panic!("unexpected baseline cmd 0x{cmd:02X}"),
            }
        }
        let mut new_cmds = Vec::new();
        new_cmds.extend_from_slice(&cmd_region[..start_idx]);
        new_cmds.extend_from_slice(&[0x07, 0x00, 0x02]); // size = 2, zero payload
        new_cmds.extend_from_slice(&cmd_region[start_idx..]);
        assert_eq!(*new_cmds.last().unwrap(), 0xFF);

        let mut out = Vec::new();
        out.extend_from_slice(&[0, 0]);
        let new_ctrl_off = 4 + rle_region.len();
        out.extend_from_slice(&(new_ctrl_off as u16).to_be_bytes());
        out.extend_from_slice(rle_region);
        let ctrl_pos = out.len();
        out.extend_from_slice(&[0, 0, 0, 0]);
        out.extend_from_slice(&new_cmds);
        out[ctrl_pos + 2] = (ctrl_pos as u16 >> 8) as u8;
        out[ctrl_pos + 3] = (ctrl_pos as u16 & 0xFF) as u8;
        let total = out.len() as u16;
        out[0] = (total >> 8) as u8;
        out[1] = (total & 0xFF) as u8;

        let (state, _, (w, h)) = parse_and_decode_spu(&out).unwrap();
        assert_eq!((w, h), (2, 2));
        assert!(state.saw_chg_colcon);
    }

    /// Helper: rebuild a 2×2 SPU with a single CHG_COLCON command,
    /// supplying a fully-formed parameter payload (including the
    /// trailing `0F FF FF FF` LN_CTLI sentinel). Caller passes the
    /// LN_CTLI-and-PX_CTLI byte sequence WITHOUT the sentinel; this
    /// helper appends it and computes the 2-byte total-size header.
    ///
    /// `bbox_x1` / `bbox_y1` are the SPU's top-left display
    /// coordinates: the test uses them to verify the band/entry
    /// coordinates are interpreted as absolute display coordinates
    /// (not bitmap-local). The 2-byte SET_DAREA encoding in
    /// `build_demo_spu` only handles `(0, 0)`-origin SPUs, so this
    /// helper rewrites the coords command in-place to point at the
    /// requested bbox.
    fn build_spu_with_chg_colcon_payload(
        width: u16,
        height: u16,
        indices: &[u8],
        bbox_x1: u16,
        bbox_y1: u16,
        payload_no_sentinel: &[u8],
    ) -> Vec<u8> {
        // Build base SPU with the requested bitmap.
        let base = build_demo_spu(width, height, indices);
        let base_ctrl = u16::from_be_bytes([base[2], base[3]]) as usize;
        let rle_region = &base[4..base_ctrl];
        let cmd_region = &base[base_ctrl + 4..];

        // Walk commands; record the (start_idx) of the START display
        // command and rewrite the SET_DAREA coords command in-place so
        // the SPU's display bbox = (bbox_x1, bbox_y1) ..= (bbox_x1 +
        // width - 1, bbox_y1 + height - 1).
        let mut new_cmd_region: Vec<u8> = cmd_region.to_vec();
        let mut walk = 0usize;
        let start_idx;
        loop {
            assert!(walk < new_cmd_region.len());
            let cmd = new_cmd_region[walk];
            match cmd {
                0x00 | 0x01 => {
                    start_idx = walk;
                    break;
                }
                0x02 => walk += 1,
                0x03 | 0x04 => walk += 1 + 2,
                0x05 => {
                    let x2 = bbox_x1 + width - 1;
                    let y2 = bbox_y1 + height - 1;
                    new_cmd_region[walk + 1] = (bbox_x1 >> 4) as u8;
                    new_cmd_region[walk + 2] =
                        (((bbox_x1 & 0x0F) as u8) << 4) | ((x2 >> 8) as u8 & 0x0F);
                    new_cmd_region[walk + 3] = (x2 & 0xFF) as u8;
                    new_cmd_region[walk + 4] = (bbox_y1 >> 4) as u8;
                    new_cmd_region[walk + 5] =
                        (((bbox_y1 & 0x0F) as u8) << 4) | ((y2 >> 8) as u8 & 0x0F);
                    new_cmd_region[walk + 6] = (y2 & 0xFF) as u8;
                    walk += 1 + 6;
                }
                0x06 => walk += 1 + 4,
                _ => panic!("unexpected baseline cmd 0x{cmd:02X}"),
            }
        }

        // Build the new command region: prefix + CHG_COLCON +
        // payload + sentinel + suffix.
        let mut payload_full = Vec::with_capacity(payload_no_sentinel.len() + 4);
        payload_full.extend_from_slice(payload_no_sentinel);
        payload_full.extend_from_slice(&[0x0F, 0xFF, 0xFF, 0xFF]);
        let size = (2 + payload_full.len()) as u16; // includes size word

        let mut new_cmds = Vec::new();
        new_cmds.extend_from_slice(&new_cmd_region[..start_idx]);
        new_cmds.push(0x07);
        new_cmds.extend_from_slice(&size.to_be_bytes());
        new_cmds.extend_from_slice(&payload_full);
        new_cmds.extend_from_slice(&new_cmd_region[start_idx..]);

        let mut out = Vec::new();
        out.extend_from_slice(&[0, 0]);
        let new_ctrl_off = 4 + rle_region.len();
        out.extend_from_slice(&(new_ctrl_off as u16).to_be_bytes());
        out.extend_from_slice(rle_region);
        let ctrl_pos = out.len();
        out.extend_from_slice(&[0, 0, 0, 0]); // delay + next placeholder
        out.extend_from_slice(&new_cmds);
        out[ctrl_pos + 2] = (ctrl_pos as u16 >> 8) as u8;
        out[ctrl_pos + 3] = (ctrl_pos as u16 & 0xFF) as u8;
        let total = out.len() as u16;
        out[0] = (total >> 8) as u8;
        out[1] = (total & 0xFF) as u8;
        out
    }

    /// Sanity-check the new bbox-rewriting branch of the helper before
    /// the more involved CHG_COLCON application tests use it.
    #[test]
    fn helper_bbox_rewrite_produces_expected_coords() {
        // Use empty payload (just sentinel) so this only validates the
        // SET_DAREA rewrite + control-block plumbing, not the CHG_COLCON
        // application logic itself.
        let spu = build_spu_with_chg_colcon_payload(2, 2, &[1u8, 1, 1, 1], 100, 50, &[]);
        let (state, _, (w, h)) = parse_and_decode_spu(&spu).unwrap();
        assert_eq!((w, h), (2, 2));
        assert_eq!((state.x1, state.y1), (100, 50));
        assert_eq!((state.x2, state.y2), (101, 51));
        assert!(state.saw_chg_colcon);
        assert!(
            state.chg_colcon.is_empty(),
            "empty payload (sentinel only) should produce zero bands"
        );
    }

    #[test]
    fn chg_colcon_parses_single_band_with_two_entries() {
        // One LN_CTLI covering display lines 10..=20 with two PX_CTLI:
        // - col 100: bg→0x5, pat→0x6, emp1→0x7, emp2→0x8;
        //            alpha bg→0x9, pat→0xA, emp1→0xB, emp2→0xC
        // - col 200: bg→0x1, pat→0x2, emp1→0x3, emp2→0x4;
        //            alpha bg→0xF, pat→0xE, emp1→0xD, emp2→0xC
        //
        // LN_CTLI: 0 sss n ttt → csln=10, n=2, ctln=20 → 00 0A 20 14
        let mut payload = Vec::new();
        payload.extend_from_slice(&[0x00, 0x0A, 0x20, 0x14]); // LN_CTLI

        payload.extend_from_slice(&100u16.to_be_bytes()); // start_col
        payload.extend_from_slice(&[0x56, 0x78]); // colors: 5/6, 7/8
        payload.extend_from_slice(&[0x9A, 0xBC]); // alphas: 9/A, B/C

        payload.extend_from_slice(&200u16.to_be_bytes());
        payload.extend_from_slice(&[0x12, 0x34]);
        payload.extend_from_slice(&[0xFE, 0xDC]);

        let spu = build_spu_with_chg_colcon_payload(2, 2, &[1u8, 1, 1, 1], 0, 0, &payload);
        let (state, _, _) = parse_and_decode_spu(&spu).unwrap();
        assert_eq!(state.chg_colcon.len(), 1);
        let band = &state.chg_colcon[0];
        assert_eq!((band.csln, band.ctln), (10, 20));
        assert_eq!(band.entries.len(), 2);
        assert_eq!(band.entries[0].start_col, 100);
        assert_eq!(band.entries[0].palette_sel, [0x5, 0x6, 0x7, 0x8]);
        assert_eq!(band.entries[0].alpha, [0x9, 0xA, 0xB, 0xC]);
        assert_eq!(band.entries[1].start_col, 200);
        assert_eq!(band.entries[1].palette_sel, [0x1, 0x2, 0x3, 0x4]);
        assert_eq!(band.entries[1].alpha, [0xF, 0xE, 0xD, 0xC]);
    }

    #[test]
    fn chg_colcon_application_mutates_canvas_alpha_inside_band() {
        // Build a 4×4 SPU at display origin (0, 0) filled with pattern
        // (RLE index 1). Base palette[1]=red, base alpha[1]=0xF
        // (opaque). CHG_COLCON band covering lines 0..=1 (top half)
        // sets alpha[1]=0 from column 0 onward.
        //
        // Result: rows 0 + 1 should be all-transparent (alpha=0),
        // rows 2 + 3 should stay solid red opaque.
        let indices = vec![1u8; 4 * 4];
        let mut payload = Vec::new();
        // LN_CTLI: csln=0, n=1, ctln=1 → 00 00 10 01
        payload.extend_from_slice(&[0x00, 0x00, 0x10, 0x01]);
        payload.extend_from_slice(&0u16.to_be_bytes()); // start_col=0
                                                        // colors: keep base (bg=0, pat=1, emp1=3, emp2=2). Byte 0
                                                        // (bg|pat) = 0x01, byte 1 (emp1|emp2) = 0x32.
        payload.extend_from_slice(&[0x01, 0x32]);
        // alphas: bg=0, pat=0 (transparent), emp1=0xF, emp2=0xF.
        payload.extend_from_slice(&[0x00, 0xFF]);

        let spu = build_spu_with_chg_colcon_payload(4, 4, &indices, 0, 0, &payload);

        // Decoder setup: palette[1]=red.
        let mut params = CodecParameters::video(CodecId::new(VOBSUB_CODEC_ID));
        let mut extra = vec![0u8; 48];
        extra[3] = 255; // palette[1] R
        params.extradata = extra;
        let mut dec = make_decoder(&params).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 1_000_000), spu).with_pts(0);
        dec.send_packet(&pkt).unwrap();
        let frame = dec.receive_frame().unwrap();
        let Frame::Video(v) = frame else {
            panic!();
        };
        // Row 0: all transparent.
        for col in 0..4 {
            let dst = col * 4;
            assert_eq!(
                v.planes[0].data[dst + 3],
                0,
                "row 0 col {col} should be transparent"
            );
        }
        // Row 1: same.
        for col in 0..4 {
            let dst = (4 + col) * 4;
            assert_eq!(
                v.planes[0].data[dst + 3],
                0,
                "row 1 col {col} should be transparent"
            );
        }
        // Row 2: solid red opaque.
        for col in 0..4 {
            let dst = (8 + col) * 4;
            assert_eq!(&v.planes[0].data[dst..dst + 4], &[255, 0, 0, 255]);
        }
        // Row 3: solid red opaque.
        for col in 0..4 {
            let dst = (12 + col) * 4;
            assert_eq!(&v.planes[0].data[dst..dst + 4], &[255, 0, 0, 255]);
        }
    }

    #[test]
    fn chg_colcon_application_respects_horizontal_start_column() {
        // 4×1 SPU at (0, 0), all pattern. Band covers row 0 only with
        // one PX_CTLI at start_col=2 making pattern transparent. Expect
        // cols 0 + 1 = red opaque, cols 2 + 3 = transparent.
        let indices = vec![1u8; 4];
        let mut payload = Vec::new();
        // LN_CTLI: csln=0, n=1, ctln=0
        payload.extend_from_slice(&[0x00, 0x00, 0x10, 0x00]);
        payload.extend_from_slice(&2u16.to_be_bytes()); // start_col=2
        payload.extend_from_slice(&[0x01, 0x32]); // colors unchanged
        payload.extend_from_slice(&[0x00, 0xFF]); // pat alpha=0

        let spu = build_spu_with_chg_colcon_payload(4, 1, &indices, 0, 0, &payload);
        let mut params = CodecParameters::video(CodecId::new(VOBSUB_CODEC_ID));
        let mut extra = vec![0u8; 48];
        extra[3] = 255;
        params.extradata = extra;
        let mut dec = make_decoder(&params).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 1_000_000), spu).with_pts(0);
        dec.send_packet(&pkt).unwrap();
        let Frame::Video(v) = dec.receive_frame().unwrap() else {
            panic!();
        };
        assert_eq!(&v.planes[0].data[0..4], &[255, 0, 0, 255]);
        assert_eq!(&v.planes[0].data[4..8], &[255, 0, 0, 255]);
        assert_eq!(v.planes[0].data[11], 0, "col 2 should be transparent");
        assert_eq!(v.planes[0].data[15], 0, "col 3 should be transparent");
    }

    #[test]
    fn chg_colcon_application_is_in_display_coords_not_bitmap_local() {
        // SPU rendered at display origin (50, 100), 4 cols × 2 rows of
        // pattern. Band covers display lines 100..=100 with one
        // PX_CTLI at display start_col=52 (i.e. bitmap-local col 2)
        // making pattern transparent.
        // Expect bitmap-local row 0 cols 0+1 opaque, cols 2+3
        // transparent; row 1 fully opaque.
        let indices = vec![1u8; 4 * 2];
        let mut payload = Vec::new();
        // LN_CTLI: csln=100, n=1, ctln=100. csln is 12-bit big-endian
        // in the low nibble of byte 0 + byte 1; ctln is 12-bit in the
        // low nibble of byte 2 + byte 3. 100 = 0x064, so:
        //   byte 0 = 0x00 (high nibble reserved zero, low nibble csln[11:8])
        //   byte 1 = 0x64 (csln[7:0])
        //   byte 2 = 0x10 (n=1, ctln[11:8]=0)
        //   byte 3 = 0x64 (ctln[7:0])
        payload.extend_from_slice(&[0x00, 0x64, 0x10, 0x64]);
        payload.extend_from_slice(&52u16.to_be_bytes()); // start_col=52
        payload.extend_from_slice(&[0x01, 0x32]);
        payload.extend_from_slice(&[0x00, 0xFF]);

        let spu = build_spu_with_chg_colcon_payload(4, 2, &indices, 50, 100, &payload);
        let mut params = CodecParameters::video(CodecId::new(VOBSUB_CODEC_ID));
        let mut extra = vec![0u8; 48];
        extra[3] = 255;
        params.extradata = extra;
        let mut dec = make_decoder(&params).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 1_000_000), spu).with_pts(0);
        dec.send_packet(&pkt).unwrap();
        let Frame::Video(v) = dec.receive_frame().unwrap() else {
            panic!();
        };
        // Row 0 (display y=100): cols 0,1 (display 50,51) opaque,
        // cols 2,3 (display 52,53) transparent.
        assert_eq!(&v.planes[0].data[0..4], &[255, 0, 0, 255]);
        assert_eq!(&v.planes[0].data[4..8], &[255, 0, 0, 255]);
        assert_eq!(v.planes[0].data[11], 0);
        assert_eq!(v.planes[0].data[15], 0);
        // Row 1 (display y=101): band doesn't cover, all opaque.
        for col in 0..4 {
            let dst = (4 + col) * 4;
            assert_eq!(&v.planes[0].data[dst..dst + 4], &[255, 0, 0, 255]);
        }
    }

    #[test]
    fn chg_colcon_payload_rejects_truncated_ln_ctli() {
        // 3 bytes — can't fit a 4-byte LN_CTLI.
        let err = parse_chg_colcon_payload(&[0x00, 0x0A, 0x10]).unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("LN_CTLI header truncated"), "got: {msg}");
    }

    #[test]
    fn chg_colcon_payload_rejects_non_zero_high_nibble() {
        // h0 high nibble must be 0 (top 4 bits of csln are reserved).
        let err = parse_chg_colcon_payload(&[0x80, 0x00, 0x10, 0x05]).unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("LN_CTLI high nibble must be zero"),
            "got: {msg}"
        );
    }

    #[test]
    fn chg_colcon_payload_rejects_inverted_lines() {
        // csln=20, ctln=10 → ctln < csln.
        let err = parse_chg_colcon_payload(&[0x00, 0x14, 0x10, 0x0A, 0x0F, 0xFF, 0xFF, 0xFF])
            .unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("ctln < csln"), "got: {msg}");
    }

    #[test]
    fn chg_colcon_payload_rejects_truncated_px_ctli() {
        // LN_CTLI announces 1 PX_CTLI but only 3 bytes of PX_CTLI follow.
        let err =
            parse_chg_colcon_payload(&[0x00, 0x00, 0x10, 0x05, 0x00, 0x00, 0x12]).unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("PX_CTLI truncated"), "got: {msg}");
    }

    #[test]
    fn chg_colcon_payload_rejects_non_increasing_px_ctli() {
        // Two PX_CTLI with start_col=10 then start_col=5.
        let mut p = Vec::new();
        p.extend_from_slice(&[0x00, 0x00, 0x20, 0x10]); // csln=0,n=2,ctln=16
        p.extend_from_slice(&10u16.to_be_bytes());
        p.extend_from_slice(&[0, 0, 0, 0]);
        p.extend_from_slice(&5u16.to_be_bytes());
        p.extend_from_slice(&[0, 0, 0, 0]);
        let err = parse_chg_colcon_payload(&p).unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("not strictly increasing"), "got: {msg}");
    }

    #[test]
    fn chg_colcon_payload_accepts_payload_without_explicit_sentinel() {
        // One LN_CTLI + one PX_CTLI, then payload ends exactly. Should
        // be accepted (some authoring tools omit the sentinel and rely
        // on the outer 2-byte size word).
        let mut p = Vec::new();
        p.extend_from_slice(&[0x00, 0x00, 0x10, 0x05]); // csln=0,n=1,ctln=5
        p.extend_from_slice(&0u16.to_be_bytes()); // start_col=0
        p.extend_from_slice(&[0x12, 0x34, 0xAB, 0xCD]);
        let bands = parse_chg_colcon_payload(&p).expect("payload-end without sentinel accepted");
        assert_eq!(bands.len(), 1);
        assert_eq!(bands[0].entries.len(), 1);
    }

    #[test]
    fn chg_colcon_lookup_last_match_wins_when_bands_overlap() {
        // Two overlapping bands; the second should win for a pixel
        // covered by both.
        let bands = vec![
            ChgColConBand {
                csln: 0,
                ctln: 10,
                entries: vec![ChgColConEntry {
                    start_col: 0,
                    palette_sel: [1, 1, 1, 1],
                    alpha: [1, 1, 1, 1],
                }],
            },
            ChgColConBand {
                csln: 5,
                ctln: 15,
                entries: vec![ChgColConEntry {
                    start_col: 0,
                    palette_sel: [2, 2, 2, 2],
                    alpha: [2, 2, 2, 2],
                }],
            },
        ];
        // y=7 is covered by both; second band wins.
        let (pal, alpha) = chg_colcon_lookup(&bands, 50, 7).unwrap();
        assert_eq!(pal, [2, 2, 2, 2]);
        assert_eq!(alpha, [2, 2, 2, 2]);
        // y=2 is covered only by band 0.
        let (pal, _) = chg_colcon_lookup(&bands, 50, 2).unwrap();
        assert_eq!(pal, [1, 1, 1, 1]);
        // y=12 is covered only by band 1.
        let (pal, _) = chg_colcon_lookup(&bands, 50, 12).unwrap();
        assert_eq!(pal, [2, 2, 2, 2]);
        // y=20 is outside both.
        assert!(chg_colcon_lookup(&bands, 50, 20).is_none());
    }

    #[test]
    fn chg_colcon_lookup_no_match_above_first_start_col() {
        // PX_CTLI starts at col 50; lookup at col 49 should miss.
        let bands = vec![ChgColConBand {
            csln: 0,
            ctln: 10,
            entries: vec![ChgColConEntry {
                start_col: 50,
                palette_sel: [3, 3, 3, 3],
                alpha: [3, 3, 3, 3],
            }],
        }];
        assert!(chg_colcon_lookup(&bands, 49, 5).is_none());
        let (pal, _) = chg_colcon_lookup(&bands, 50, 5).unwrap();
        assert_eq!(pal, [3, 3, 3, 3]);
        let (pal, _) = chg_colcon_lookup(&bands, 9999, 5).unwrap();
        assert_eq!(pal, [3, 3, 3, 3]);
    }

    #[test]
    fn chg_colcon_application_random_no_panic_sweep() {
        // 200 iterations: random bitmap + random in-bounds CHG_COLCON
        // band/entries. Must never panic and must produce a canvas of
        // the expected size.
        let mut rng = 0xC0FFEEu32;
        let mut lcg = || {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            rng
        };
        for _ in 0..200 {
            let w = 1 + (lcg() % 8) as u16;
            let h = 1 + (lcg() % 8) as u16;
            let pixels: Vec<u8> = (0..(w as usize) * (h as usize))
                .map(|_| (lcg() & 0x03) as u8)
                .collect();
            // Build a CHG_COLCON payload with 1..=3 LN_CTLI bands, each
            // with 1..=2 PX_CTLI entries with strictly-increasing
            // start_col values.
            let bbox_x1 = 0;
            let bbox_y1 = 0;
            let bbox_x2 = bbox_x1 + w - 1;
            let bbox_y2 = bbox_y1 + h - 1;
            let mut payload = Vec::new();
            let nb = 1 + (lcg() % 3);
            for _ in 0..nb {
                let csln = bbox_y1 + (lcg() % h as u32) as u16;
                let ctln_lo = csln;
                let ctln_hi = bbox_y2;
                let ctln = ctln_lo + (lcg() % (ctln_hi - ctln_lo + 1) as u32) as u16;
                let ne = 1 + (lcg() % 2);
                let mut starts: Vec<u16> = Vec::new();
                let mut sc = bbox_x1;
                for _ in 0..ne {
                    if sc > bbox_x2 {
                        break;
                    }
                    starts.push(sc);
                    sc = sc.saturating_add(1 + (lcg() % 3) as u16);
                }
                if starts.is_empty() {
                    continue;
                }
                payload.extend_from_slice(&[
                    (csln >> 8) as u8 & 0x0F,
                    (csln & 0xFF) as u8,
                    (((starts.len() as u8) & 0x0F) << 4) | ((ctln >> 8) as u8 & 0x0F),
                    (ctln & 0xFF) as u8,
                ]);
                for s in &starts {
                    payload.extend_from_slice(&s.to_be_bytes());
                    payload.extend_from_slice(&[
                        (lcg() & 0xFF) as u8,
                        (lcg() & 0xFF) as u8,
                        (lcg() & 0xFF) as u8,
                        (lcg() & 0xFF) as u8,
                    ]);
                }
            }
            if payload.is_empty() {
                continue;
            }
            let spu = build_spu_with_chg_colcon_payload(w, h, &pixels, bbox_x1, bbox_y1, &payload);
            let mut params = CodecParameters::video(CodecId::new(VOBSUB_CODEC_ID));
            params.extradata = vec![0u8; 48];
            let mut dec = make_decoder(&params).unwrap();
            let pkt = Packet::new(0, TimeBase::new(1, 1_000_000), spu).with_pts(0);
            if dec.send_packet(&pkt).is_err() {
                continue; // malformed coords / band layout; OK to reject
            }
            let frame = dec.receive_frame().unwrap();
            let Frame::Video(v) = frame else {
                panic!();
            };
            assert_eq!(
                v.planes[0].data.len(),
                (w as usize) * (h as usize) * 4,
                "canvas size mismatch"
            );
        }
    }

    /// Build an SPU whose control-sequence area is a chain of DCSQs the
    /// caller spells out as `(stm, commands)` pairs. The RLE bitmap is a
    /// fixed 2×2 paint of palette index 1 (taken from the standard demo
    /// builder); this helper exists to exercise the control-sequence /
    /// DCSQ-traversal logic, not RLE coverage. Each `commands` slice is
    /// the command-bytes payload between the 4-byte DCSQ header (STM +
    /// next-pointer) and the next DCSQ; callers are responsible for
    /// terminating with `0xFF` themselves where appropriate. The final
    /// DCSQ in the chain has its next-pointer rewritten to point at
    /// itself, matching the spec's "if this is the last SP_DCSQ, it
    /// points to itself" rule.
    fn build_spu_with_dcsq_chain(dcsqs: &[(u16, &[u8])]) -> Vec<u8> {
        // Borrow the demo's RLE bytes verbatim so we have a known-good
        // SET_COLOR / SET_CONTR / SET_DAREA / SET_DSPXA wired up before
        // the caller's DCSQs run. The demo packs top-field bytes first
        // and bottom-field bytes second; we recover both offsets by
        // pulling them straight out of the demo's SET_DSPXA (command
        // 0x06) so the relative positions inside our copy of the RLE
        // region stay valid even if the demo's RLE size changes.
        let base = build_demo_spu(2, 2, &[1u8, 1, 1, 1]);
        let base_ctrl = u16::from_be_bytes([base[2], base[3]]) as usize;
        let rle = base[4..base_ctrl].to_vec();
        let (base_top_off, base_bot_off) = {
            // Walk the base's first (and only) DCSQ commands looking for
            // the SET_DSPXA opcode, then read its 4-byte argument.
            let walk_from = base_ctrl + 4;
            let mut w = walk_from;
            let (mut top, mut bot) = (4u16, 4u16);
            while w < base.len() {
                let cmd = base[w];
                w += 1;
                match cmd {
                    0x00..=0x02 => {}
                    0x03 | 0x04 => w += 2,
                    0x05 => w += 6,
                    0x06 => {
                        top = u16::from_be_bytes([base[w], base[w + 1]]);
                        bot = u16::from_be_bytes([base[w + 2], base[w + 3]]);
                        w += 4;
                    }
                    0xFF => break,
                    _ => unreachable!("base demo SPU should only use known commands"),
                }
            }
            (top, bot)
        };

        // The first DCSQ carries the standard palette / alpha / coords /
        // RLE-offsets so the rest of `parse_and_decode_spu` runs end to
        // end. Build its body once. Our RLE region starts at the same
        // offset (4) as the base's, so the offsets are reused verbatim.
        let top_off = base_top_off;
        let bot_off = base_bot_off;
        let mut first_body: Vec<u8> = vec![
            0x03, // SET_COLOR
            0x01, // bg=0, pat=1
            0x32, // emp1=3, emp2=2
            0x04, // SET_CONTR
            0x0F, // bg=0, pat=0xF
            0xFF, // emp1=0xF, emp2=0xF
            0x05, // SET_DAREA — coords (x1=0, x2=1, y1=0, y2=1)
            0, 0, 1, 0, 0, 1,    //
            0x06, // SET_DSPXA — top/bottom RLE offsets
        ];
        first_body.extend_from_slice(&top_off.to_be_bytes());
        first_body.extend_from_slice(&bot_off.to_be_bytes());
        first_body.push(0xFF); // CMD_END (this DCSQ ends without an STA_DSP)

        // Assemble the SPU header + RLE + DCSQ chain.
        let mut out: Vec<u8> = Vec::new();
        out.extend_from_slice(&[0, 0]); // placeholder SPU length
        out.extend_from_slice(&[0, 0]); // placeholder ctrl offset
        out.extend_from_slice(&rle);
        let ctrl_off = out.len();
        out[2] = (ctrl_off >> 8) as u8;
        out[3] = (ctrl_off & 0xFF) as u8;

        // First record the offset of every DCSQ so we can write
        // accurate next-pointers in a second pass.
        let mut dcsq_offsets: Vec<usize> = Vec::with_capacity(1 + dcsqs.len());

        // Emit the implicit setup DCSQ first.
        dcsq_offsets.push(out.len());
        out.extend_from_slice(&[0, 0]); // STM = 0
        out.extend_from_slice(&[0, 0]); // next placeholder
        out.extend_from_slice(&first_body);

        // Then each caller-supplied DCSQ.
        for (stm, body) in dcsqs {
            dcsq_offsets.push(out.len());
            out.extend_from_slice(&stm.to_be_bytes());
            out.extend_from_slice(&[0, 0]); // next placeholder
            out.extend_from_slice(body);
        }

        // Patch every DCSQ's next-pointer to the *following* DCSQ's
        // offset, except the last which points to itself.
        for (i, &dcsq_pos) in dcsq_offsets.iter().enumerate() {
            let next_pos = if i + 1 < dcsq_offsets.len() {
                dcsq_offsets[i + 1]
            } else {
                dcsq_pos
            };
            let next_u16 = next_pos as u16;
            out[dcsq_pos + 2] = (next_u16 >> 8) as u8;
            out[dcsq_pos + 3] = (next_u16 & 0xFF) as u8;
        }

        // Finalise the outer SPU length.
        let total = out.len() as u16;
        out[0] = (total >> 8) as u8;
        out[1] = (total & 0xFF) as u8;
        out
    }

    #[test]
    fn dcsq_chain_helper_produces_decodable_setup_only_spu() {
        // Sanity check: with an empty chain, the helper still produces a
        // single setup DCSQ whose CMD_END terminates cleanly, and the
        // SPU decodes its 2×2 bitmap as before. No STA_DSP appears, so
        // start_latched stays false and `start_delay_raw` remains 0.
        let spu = build_spu_with_dcsq_chain(&[]);
        let (state, pixels, (w, h)) = parse_and_decode_spu(&spu).unwrap();
        assert_eq!((w, h), (2, 2));
        assert_eq!(pixels, vec![1, 1, 1, 1]);
        assert_eq!(state.start_delay_raw, 0);
        assert_eq!(state.stop_delay_raw, 0);
        assert!(!state.forced_display);
    }

    #[test]
    fn sta_dsp_latches_delay_from_owning_dcsq_not_first() {
        // Realistic SPU shape: the *first* DCSQ does setup with STM=0
        // and *no* STA_DSP, and a later DCSQ schedules STA_DSP with a
        // non-zero STM (the delay before the cue appears). The latched
        // `start_delay_raw` must be the second DCSQ's STM, not the
        // first DCSQ's (which would always be zero on this shape and
        // therefore useless as a scheduling signal).
        let spu = build_spu_with_dcsq_chain(&[(0x1234, &[0x01, 0xFF])]);
        let (state, _, _) = parse_and_decode_spu(&spu).unwrap();
        assert_eq!(
            state.start_delay_raw, 0x1234,
            "start_delay_raw must come from the STA_DSP-bearing DCSQ"
        );
        assert!(
            !state.forced_display,
            "STA_DSP alone must not set the forced_display flag"
        );
    }

    #[test]
    fn fsta_dsp_sets_forced_display_and_latches_delay() {
        // FSTA_DSP (0x00) is the spec's forced-start opcode — the SPU
        // asks the player to display the cue even when subtitles are
        // disabled. The flag captures presence, and the same
        // first-encountered-start latching rule applies to the DCSQ's
        // STM as for STA_DSP.
        let spu = build_spu_with_dcsq_chain(&[(0x00AB, &[0x00, 0xFF])]);
        let (state, _, _) = parse_and_decode_spu(&spu).unwrap();
        assert!(state.forced_display, "FSTA_DSP must set forced_display");
        assert_eq!(
            state.start_delay_raw, 0x00AB,
            "FSTA_DSP must also latch start_delay_raw"
        );
    }

    #[test]
    fn stp_dsp_overwrites_with_latest_dcsq_stm() {
        // STP_DSP differs from STA_DSP in that the spec text reads the
        // latest stop-display as the authoritative one. Multiple stops
        // across separate DCSQs end up reporting the final STM.
        let spu = build_spu_with_dcsq_chain(&[(0x0010, &[0x02, 0xFF]), (0x0050, &[0x02, 0xFF])]);
        let (state, _, _) = parse_and_decode_spu(&spu).unwrap();
        assert_eq!(
            state.stop_delay_raw, 0x0050,
            "later STP_DSP must overwrite earlier stop delay"
        );
    }

    #[test]
    fn first_start_wins_when_two_dcsqs_both_assert_sta_dsp() {
        // The first start-display command encountered in DCSQ traversal
        // order wins; a later DCSQ's STA_DSP is a redundant retrigger
        // and does not overwrite the latched delay.
        let spu = build_spu_with_dcsq_chain(&[(0x0030, &[0x01, 0xFF]), (0x0099, &[0x01, 0xFF])]);
        let (state, _, _) = parse_and_decode_spu(&spu).unwrap();
        assert_eq!(
            state.start_delay_raw, 0x0030,
            "first STA_DSP's DCSQ STM must be the latched start delay"
        );
    }

    #[test]
    fn fsta_dsp_after_sta_dsp_still_sets_forced_flag_without_relatching_delay() {
        // If an SPU somehow contains both a regular STA_DSP first and a
        // forced FSTA_DSP later, the `forced_display` flag still
        // surfaces (it captures presence anywhere in the chain), and
        // the `start_delay_raw` stays at the first-encountered start
        // command's DCSQ STM. This shape isn't common but we shouldn't
        // silently lose the forced annotation.
        let spu = build_spu_with_dcsq_chain(&[(0x0010, &[0x01, 0xFF]), (0x0080, &[0x00, 0xFF])]);
        let (state, _, _) = parse_and_decode_spu(&spu).unwrap();
        assert_eq!(state.start_delay_raw, 0x0010, "first start wins");
        assert!(
            state.forced_display,
            "FSTA_DSP anywhere in the chain must set forced_display"
        );
    }

    #[test]
    fn self_referential_terminator_does_not_spin() {
        // The mpucoder SPU spec says the last DCSQ's
        // `SP_NXT_DCSQ_SA` points at itself — the helper does that on
        // every chain. Even with a long chain we must reach the end in
        // bounded time. (Without the `next <= pos` guard this loops
        // forever.)
        let bodies: Vec<&[u8]> = (0..16).map(|_| &[0x01u8, 0xFFu8][..]).collect();
        let chain: Vec<(u16, &[u8])> = bodies.iter().map(|b| (0x0001u16, *b)).collect();
        let spu = build_spu_with_dcsq_chain(&chain);
        // Termination check is implicit: this returning at all means we
        // didn't spin on the self-pointer of the final DCSQ.
        let _ = parse_and_decode_spu(&spu).unwrap();
    }

    #[test]
    fn unknown_command_in_later_dcsq_errors_not_panics() {
        // Verify the unknown-command escape still produces a tidy
        // Error::invalid even when the unknown opcode lives in a
        // *non-first* DCSQ — this exercises the path now that
        // `first_seq` is gone.
        let spu = build_spu_with_dcsq_chain(&[(0x0010, &[0xAB, 0xFF])]);
        let err = parse_and_decode_spu(&spu).unwrap_err();
        let s = format!("{}", err);
        assert!(
            s.contains("unknown command"),
            "unexpected error variant: {s}"
        );
    }

    #[test]
    fn extracts_spu_from_mpeg_ps_wrap() {
        // Build one SPU, wrap it into a minimal MPEG-2 PS with pack +
        // one private_stream_1 PES packet carrying the DVD substream id.
        let spu = build_demo_spu(2, 2, &[1u8, 1, 1, 1]);
        let mut ps = Vec::new();
        // Pack header (MPEG-2): 0x000001BA, 10 bytes incl. 0 stuffing.
        ps.extend_from_slice(&[0, 0, 1, 0xBA]);
        // byte 0 starts with 01 → MPEG-2 marker
        ps.extend_from_slice(&[0x44, 0x00, 0x04, 0x00, 0x04, 0x01, 0x00, 0x00, 0x00, 0xF8]);
        // Private stream 1 PES: start code + 2-byte length + PES hdr +
        // substream id (0x20) + SPU payload.
        let substream = 0x20u8;
        let pes_payload_len = 3 + 1 + spu.len(); // flags(2) + hdr_len(1) + substream + spu
        ps.extend_from_slice(&[0, 0, 1, 0xBD]);
        ps.extend_from_slice(&(pes_payload_len as u16).to_be_bytes());
        // Minimal MPEG-2 PES header: 0x80, 0x00, 0x00 (no PTS/DTS, 0 hdr data).
        ps.extend_from_slice(&[0x80, 0x00, 0x00]);
        ps.push(substream);
        ps.extend_from_slice(&spu);

        let out = extract_spu(&ps).unwrap();
        assert_eq!(out, spu);
    }
}
