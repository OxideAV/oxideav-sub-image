# oxideav-sub-image

Pure-Rust bitmap-subtitle codecs and containers:

- **PGS** (HDMV / Blu-ray `.sup`) — decode + encode, standalone `.sup`
  container.
- **DVB subtitles** (ETSI EN 300 743) — decode + encode, no standalone
  container (DVB subs ride inside MPEG-TS; the encoder emits the
  display-set PES payload, so getting on air still needs a TS muxer
  upstream). The write direction covers every segment the decoder
  reads: `write_segment` framing plus display-definition,
  page-composition, region-composition (fill flag, per-depth fill
  pixel codes, object placements), CLUT-definition (4-byte full-range
  and packed 2-byte entry forms, with `rgba_to_clut_ycbcrt` as the
  integer inverse of the decode-side BT.601 transform — greys are
  bit-exact, colours within ±2 LSBs, alpha exact), and object-data
  writers, the last with true run-length 2/4/8-bit pixel-code-string
  encoders (all counted-run forms, single-pixel-of-colour-0 codes,
  end-of-line and end-of-string markers, byte alignment) and optional
  2-byte map-table prefixes. `make_encoder` quantises RGBA frames into
  a CLUT (≤ 4 entries → 2-bit, ≤ 16 → 4-bit, else 8-bit; > 255
  distinct colours fall back to 3-3-2-2 channel reduction), crops to
  the tight bounding box of non-transparent pixels, and emits one
  complete display set per frame; a fully-transparent frame becomes an
  erase page referencing no regions. Every writer is pinned by
  encode→decode roundtrips through the existing decode path,
  including max-run / chunked-over-max / boundary-length edge cases
  for each run-length form.

  On the decode side, the decoder implements the **epoch state machine**
  (§5.1 / §5.2 / §7.2.2): the composition buffer (page composition, region
  compositions, CLUT definitions) and the pixel buffer (object data)
  persist across PES packets within an epoch rather than being rebuilt per
  packet. The page-composition `page_state` (Table 3) drives it — "mode
  change" discards all retained state before applying the display set (a
  new epoch), while "acquisition point" and "normal case" accumulate onto
  it. This is what makes a real broadcast stream render: CLUTs / regions /
  objects are sent once at mode-change and subsequent "normal case"
  display sets re-send only the deltas (often just the page composition,
  or a CLUT to recolour a retained object), which a stateless decoder
  would paint blank. `reset()` (seek) ends the epoch and drops the
  retained buffers; the encoder flags every self-contained set it emits as
  "mode change" so an IRD can acquire on any frame.

  CLUT entries follow §7.2.4 colorimetry — BT.601 Y/Cr/Cb plus T
  (alpha = 255 − T) — including the **`Y_value == 0` full-transparency**
  rule: a Y=0 entry decodes to alpha 0 regardless of its T / chroma
  fields ("Full transparency is acquired through a value of zero in the
  Y_value field"). To keep a visible colour off that sentinel, the encoder
  floors any opaque entry's luma to the §7.2.4 NOTE 1 legal value (Y=16),
  so opaque pure black round-trips to a near-black grey rather than a
  transparent pixel (DVB cannot represent opaque black).

  On the decode side, pixel-data sub-block **map-tables** (§7.2.5.1) are
  applied rather than skipped: a 2-bit or 4-bit pixel-code string carried
  inside a deeper region is remapped onto the region CLUT entry numbers
  through the active map-table. The `0x20`/`0x21`/`0x22` sub-blocks
  redefine the table in place; absent an explicit redefinition the table
  holds its §10.4–10.6 default contents (2→4 `{0,7,8,15}`, 2→8
  `{0x00,0x77,0x88,0xFF}`, 4→8 nibble-replicated). Whether a 2-bit string
  uses the 2→4 or 2→8 default, and whether a remap happens at all, is
  driven by the referencing region's colour depth (Figure 8): a 2-bit
  region uses 2-bit codes directly, a depth-≤4 region uses 4-bit codes
  directly, and 8-bit codes always index the CLUT directly. Objects are
  decoded lazily per referencing region so the correct depth drives the
  remap, and the write side emits map-tables of the correct on-wire size
  (2/4/16 bytes for `0x20`/`0x21`/`0x22`).

  Character-coded objects (`object_coding_method == 0x01`, §7.2.4) are
  parsed rather than rejected: the object-data body's `number_of_codes`
  byte is followed by that many 16-bit `character_code` values, each an
  index into the character table named in the subtitle_descriptor, and
  those are surfaced on the decoded object. The pairing region-object
  loop reads the `object_type` (Table 6: bitmap / character / composite
  character string) and, for the two character types, the
  `foreground_pixel_code`/`background_pixel_code` 8-bit-CLUT entries the
  encoder selected — advancing past them so a following object stays
  aligned. Because §5.4.6 states the stream alone "is not sufficient to
  make such a character coded system work reliably" and defers glyph
  size/metrics/rasterisation to a local broadcaster/IRD font agreement,
  the codes are carried for a caller holding that agreement and the
  object paints nothing on the canvas, rather than the decoder
  fabricating a font. Reserved coding methods (0x02/0x03), a truncated
  `character_code` list, and a character region object missing its
  fg/bg bytes are all rejected.

  The region composition
  segment's `region_fill_flag` is honoured: when set, the region
  rectangle is pre-painted with the depth-appropriate
  `region_n-bit_pixel_code` (8/4/2-bit, translated through the
  region's CLUT) *before* any objects composite on top, matching the
  spec's "fill the region area with the colour given by the n-bit
  pixel code" wording. Cleared, the rectangle stays at the canvas's
  transparent background. The pre-fill is clipped to the canvas, so a
  region declared past the right or bottom edge writes only its
  in-bounds intersection. A display set that references several regions
  composites them in page-composition list order: each region paints
  over whatever an earlier region already wrote (Porter–Duff
  source-over), so a later region's partially-transparent CLUT entries
  show the region beneath through rather than discarding it.

  An object's `non_modifying_colour_flag` (§7.2.5) is honoured: when set,
  CLUT index 1 is the *non-modifying colour*, so any pixel carrying that
  index leaves the underlying region background / lower-z-order object
  untouched instead of painting CLUT entry 1 — the spec's mechanism for
  punching "transparent holes" through an object. Cleared, index 1 paints
  normally. The flag is decoded from the object-data version/coding byte
  and the write side can emit it via `write_object_data_flags`.

  The Display Definition Segment's `display_window_flag` (§7.2.1) is
  parsed: when set, the four inclusive `display_window_*_position_
  minimum/maximum` fields are read into a typed `DisplayWindow` and the
  composited canvas is confined to that rectangle — region addresses
  stay absolute display coordinates, so anything rendered outside the
  window is cleared to the transparent background, matching the spec's
  "intended to be rendered in a window within the display size." A
  window-flagged body truncated below 13 bytes, or one whose maximum
  falls below its minimum, is rejected. `dds_version_number` is decoded
  for callers that track per-segment versioning. The write side mirrors
  this through `write_display_definition_windowed`.

  The Disparity Signalling Segment (DSS, segment type `0x15`, §7.2.7)
  is parsed for plano-stereoscopic (3D) placement. `parse_disparity_signalling`
  returns a typed `DisparitySignalling`: the modulo-16
  `dss_version_number`, the signed `page_default_disparity_shift`
  applied when a decoder can only carry one disparity per page, an
  optional page-level `DisparityShiftUpdateSequence`, and the region
  loop. Each `DisparityRegion` carries one to four `DisparitySubregion`s
  whose positional `subregion_horizontal_position`/`subregion_width`
  are present only when a region declares more than one subregion — a
  single implicit subregion reports them as `None` and spans the whole
  region. The unsigned 4-bit fractional disparity is exposed as
  1/16-pixel units to be *added* to the signed integer part (the spec's
  `-0.75 ⇒ [-1, 4/16]` convention). `DisparityShiftUpdateSequence`
  decodes the 24-bit `interval_duration` and each division period's
  `interval_count` + signed integer update, for both the page-flag and
  region-flag forms. The decoder routes the DSS through its segment
  loop and validates it, but the painted RGBA canvas stays the
  disparity-zero (2D) baseline view the spec mandates as the implicit
  default, so the parsed values are surfaced for stereoscopic callers
  rather than shifting the emitted picture. Truncated update-sequence
  or subregion bodies are rejected.
- **VobSub** / DVD SPU (`.idx`+`.sub`) — decode only, container reads
  the `.idx` text index + matched `.sub` payload (MPEG-PS pack + PES
  private_stream_1 or raw SPU-length-prefixed form). The SPU's
  Display Control Sequence chain is walked end-to-end; each DCSQ
  contributes its `SP_DCSQ_STM` (delay before its commands take
  effect) to whichever event-trigger command appears inside it, not
  to the first DCSQ unconditionally. `STA_DSP` (`0x01`) latches
  `Spu::start_delay_raw` to *its* DCSQ's STM the first time it is
  seen — typical streams put setup (palette / alpha / coords /
  `SET_DSPXA`) in DCSQ #0 with `STM = 0` and schedule the on-display
  event in DCSQ #1 with a non-zero STM, so reading the first DCSQ
  would have permanently reported a zero start delay. `FSTA_DSP`
  (`0x00`) follows the same latching rule and additionally raises
  `Spu::forced_display`, which surfaces the spec's *Forced Start
  Display* flag (subtitle a player should show even when subtitles
  are otherwise disabled — typically used for on-screen-sign
  translations). `STP_DSP` (`0x02`) writes `Spu::stop_delay_raw`
  unconditionally on every occurrence so that an authoring tool's
  revised end time inside a later DCSQ wins. The traversal exits
  when a DCSQ's `SP_NXT_DCSQ_SA` does not advance the cursor — the
  spec's "if this is the last SP_DCSQ, it points to itself"
  terminator.

  Mid-display `CHG_COLCON` palette / contrast change commands
  (opcode `0x07`) are parsed into structured `LN_CTLI`
  (vertically-bounded band) + `PX_CTLI` (horizontal start-column
  transition) entries and **applied** to the rendered bitmap: each
  pixel inside a band's `csln..=ctln` line range picks up the
  right-most matching `PX_CTLI`'s replacement 4-entry palette /
  4-entry alpha in lieu of the SPU's base `SET_COLOR` / `SET_CONTR`
  selections, running rightward until the next `PX_CTLI` or the
  right edge of the display area. Coordinates are absolute
  display-line / display-column space and are intersected with the
  SPU's bounding box. The parser tolerates payloads with or without
  the explicit `0F FF FF FF` `LN_CTLI` sentinel; it rejects
  truncated `LN_CTLI` / `PX_CTLI`, a non-zero reserved high nibble,
  `ctln < csln`, and a non-strictly-increasing `start_col` inside a
  band. `Spu::saw_chg_colcon` still surfaces that the command was
  present, and the parsed bands are exposed on `Spu::chg_colcon`
  for callers that need them.

All three decoders emit RGBA `oxideav_core::VideoFrame` values — one
frame per display-set. The stream's media kind is `Subtitle` even though
the frame kind is `Video` — the dual tagging surfaces the
bitmap-subtitle nature of these codecs to downstream consumers (player,
mixer, file writer) without losing the fact that what arrives is a
pre-rendered RGBA picture.

The PGS demuxer classifies each emitted display-set against the PCS
`composition_state` byte and flags `Packet::flags.keyframe` only when
the set is a real random-access point (`Acquisition Point` or
`Epoch Start`). Mid-epoch `Normal Case` updates depend on palette /
object state carried by an earlier set and stay un-flagged, so seekers
that round to the nearest keyframe land on a set that decodes
standalone. The parsed `composition_state`, `palette_update_flag`, and
`palette_id` fields are surfaced on `PresentationComposition`; the
`COMP_STATE_NORMAL` / `COMP_STATE_ACQUISITION` / `COMP_STATE_EPOCH_START`
constants and the `is_random_access(state)` helper let downstream
consumers act on the classification without inspecting raw bytes.

Mid-epoch **palette-only updates** are applied rather than rendered blank.
The decoder retains object / window / palette state across display-sets
inside an epoch, so a Normal-Case PCS whose `palette_update_flag` byte is
`0x80` ("palette-only update") re-renders the prior set's composition
against the freshly-merged palette — the mechanism a BD-ROM authoring tool
uses for fade-in / fade-out and colour-change effects, where a single
object is painted once and then re-coloured over successive display-sets
without re-sending the bitmap. Such a PCS reuses the previous set's
composition objects when it declares none of its own (and uses its own
when it re-lists them); either way the retained ODS object buffer supplies
the graphics. An `Epoch Start` PCS resets the retained objects, windows and
palettes so a new epoch starts clean, and a decoder `reset()` (issued on
seek) discards the retained epoch so a post-seek palette update can't
resurrect pre-seek graphics. A packet carrying no control segment of its
own still presents nothing.

The Window Definition Segment is parsed into a typed
`Vec<WindowDefinition>` (exposed via the `parse_wds` helper) and the
decoder retains the table keyed by `window_id`. Each composition object
references one window, and that window is the only canvas region the
object is permitted to paint into — at render time the planned paint
rectangle is intersected with the matching window, and any source
pixels that would land outside get trimmed before the blit. Objects
that land entirely outside their window paint nothing, and a
display-set whose WDS declares zero windows (the canonical "erase"
form) keeps the prior whole-canvas paint area for backward
compatibility. The parser rejects a body whose declared window count
does not match the remaining bytes exactly, a zero-extent window, and
an empty body.

Palette Definition Segments are kept in independent slots keyed by
their on-wire `palette_id`, so a display-set carrying several PDS for
different ids (the BD-ROM HDMV authoring shape for colour-change /
fade effects, where a content tool writes both halves of the effect
side-by-side and lets the PCS pick which one is current) no longer
collapses into one shared table. The render path consults the PCS's
`palette_id` byte and looks the matching slot up at composite time; a
PCS that references a slot no PDS has populated falls back to the
default all-transparent palette so a malformed stream stays defined.
A repeated PDS for an already-populated slot still adds / replaces
individual entries on top of whatever the earlier PDS wrote, matching
the per-entry-delta rule a `palette_version_number` bump implies.

When a display-set stacks overlapping objects (multiple PGS composition
objects, multiple DVB regions/objects) and the topmost pixel is only
partially transparent, the painted source is alpha-composited *over* the
existing canvas content using the Porter–Duff source-over operator
(`composite::over` / `composite::blit_indexed`), rather than hard-copying
and discarding what's underneath. Fully-opaque and fully-transparent
pixels short-circuit to a plain copy / no-op.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone. Zero C dependencies.

## Installation

```toml
[dependencies]
oxideav-core = "0.1"
oxideav-codec = "0.1"
oxideav-container = "0.1"
oxideav-sub-image = "0.0"
```

## Quick use

### Decoding a `.sup` file

```rust
use oxideav_core::{Frame, RuntimeContext};

let mut ctx = RuntimeContext::new();
oxideav_sub_image::register(&mut ctx);
let codecs = &ctx.codecs;
let containers = &ctx.containers;

let input: Box<dyn oxideav_container::ReadSeek> = Box::new(
    std::io::Cursor::new(std::fs::read("subs.sup")?),
);
let mut dmx = containers.open("pgs", input)?;
let stream = &dmx.streams()[0];
let mut dec = codecs.make_decoder(&stream.params)?;

loop {
    match dmx.next_packet() {
        Ok(pkt) => {
            dec.send_packet(&pkt)?;
            while let Ok(Frame::Video(vf)) = dec.receive_frame() {
                // vf.format == PixelFormat::Rgba
                // vf.planes[0].data is the composed subtitle canvas.
            }
        }
        Err(oxideav_core::Error::Eof) => break,
        Err(e) => return Err(e.into()),
    }
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

### Encoding PGS

```rust
use oxideav_core::{CodecId, CodecParameters, Frame, MediaType, PixelFormat};

let mut params = CodecParameters::video(CodecId::new("pgs"));
params.media_type = MediaType::Subtitle;
params.pixel_format = Some(PixelFormat::Rgba);

let mut enc = codecs.make_encoder(&params)?;
enc.send_frame(&Frame::Video(rgba_frame))?;
let packet = enc.receive_packet()?;
// packet.data is one complete PGS display-set (PCS + WDS + PDS + ODS + END).
```

Each `send_frame` call produces one packet. The encoder first finds the
tight bounding box of the frame's non-transparent pixels and emits one
composition object covering just that sub-rectangle at the bbox's
`(x, y)`; the surrounding transparent area is not encoded. Colour is
then quantised into a ≤ 255-entry palette (index 0 is reserved for
fully-transparent). When the bbox has more than 255 distinct RGBA
colours the encoder falls back to a 3/3/2/2 (R/G/B/A) bucketed
reduction; otherwise the quantisation is lossless up to the BT.601
RGB↔YCbCr round-trip PGS does in the palette. A fully-transparent
input frame emits an erase display-set (PCS with zero composition
objects + empty WDS), which the decoder maps to a fully-transparent
canvas — the canonical way to clear whatever was on screen before.

When the encoded object's run-length data exceeds the 65535-byte limit
of the PG segment header's 16-bit `segment_size` field — a large,
heavily-antialiased caption whose RLE defeats run-length compression —
the encoder **fragments the object across multiple ODS segments** sharing
one `object_id`, with the `last_in_sequence_flag` first/last bits set per
the wire format and only the first fragment carrying the
`object_data_length` + `width` + `height` header. The decoder reassembles
the fragments by `object_id` before interpreting them, so a large frame
round-trips pixel-exact. Previously the single-ODS body length truncated
through the `as u16` cast and corrupted the stream.

### Codec / container IDs

| Codec id  | Container id | Extensions    | Encode |
|-----------|--------------|---------------|--------|
| `pgs`     | `pgs`        | `.sup`        | yes    |
| `dvbsub`  | *(none)*     | *(TS only)*   | no     |
| `vobsub`  | `vobsub`     | `.idx`+`.sub` | no     |

### Why PGS encode only?

DVB subs are carried inside MPEG-TS PES packets, so emitting a valid
standalone stream requires a TS-aware muxer upstream. VobSub similarly
requires the `.idx` text index and `.sub` binary to be written in
lock-step, which this crate doesn't yet expose through the encoder
trait. Both decoders read the preformatted payloads the container/TS
demuxer delivers; adding encoders waits on the container layer growing
the muxer counterparts.

## License

MIT — see [LICENSE](LICENSE).
