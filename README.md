# oxideav-sub-image

Pure-Rust bitmap-subtitle codecs and containers:

- **PGS** (HDMV / Blu-ray `.sup`) — decode + encode, standalone `.sup`
  container.
- **DVB subtitles** (ETSI EN 300 743) — decode only, no standalone
  container (DVB subs ride inside MPEG-TS).
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
