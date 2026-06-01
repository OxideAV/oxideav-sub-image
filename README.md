# oxideav-sub-image

Pure-Rust bitmap-subtitle codecs and containers:

- **PGS** (HDMV / Blu-ray `.sup`) — decode + encode, standalone `.sup`
  container.
- **DVB subtitles** (ETSI EN 300 743) — decode only, no standalone
  container (DVB subs ride inside MPEG-TS).
- **VobSub** / DVD SPU (`.idx`+`.sub`) — decode only, container reads
  the `.idx` text index + matched `.sub` payload (MPEG-PS pack + PES
  private_stream_1 or raw SPU-length-prefixed form). Mid-display
  `CHG_COLCON` palette / contrast change commands (opcode `0x07`) are
  length-skipped — their rectangular palette / alpha mutations are not
  applied to the rendered bitmap, but streams that carry the command
  decode their base palette / alpha / RLE successfully and surface the
  request via `Spu::saw_chg_colcon`.

All three decoders emit RGBA `oxideav_core::VideoFrame` values — one
frame per display-set. The stream's media kind is `Subtitle` even though
the frame kind is `Video` — the dual tagging surfaces the
bitmap-subtitle nature of these codecs to downstream consumers (player,
mixer, file writer) without losing the fact that what arrives is a
pre-rendered RGBA picture.

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
