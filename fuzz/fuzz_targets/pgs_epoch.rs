#![no_main]

//! PGS multi-packet epoch fuzz target.
//!
//! Drives one decoder across a *sequence* of fuzzer-sliced packets so the
//! retained-state paths are exercised: the composition / window / palette /
//! object buffers that persist across display-sets within an epoch, the
//! palette-only-update re-render, the Epoch-Start reset, and an occasional
//! `reset()` (the seek path). A single-packet target can't reach the
//! cross-packet state machine that makes a real broadcast/disc stream
//! render.
//!
//! Wire framing: the input is a series of length-prefixed chunks — one
//! leading byte gives a chunk length (capped), the next that-many bytes are
//! one packet's payload, repeated until the input is consumed. A chunk
//! length byte with the high bit set additionally triggers a `reset()`
//! before that packet.
//!
//! Contract: **no-panic** across the whole sequence.

use libfuzzer_sys::fuzz_target;
use oxideav_core::{CodecId, CodecParameters, MediaType, Packet, PixelFormat, TimeBase};
use oxideav_sub_image::{pgs, PGS_CODEC_ID};

fn params() -> CodecParameters {
    let mut p = CodecParameters::video(CodecId::new(PGS_CODEC_ID));
    p.media_type = MediaType::Subtitle;
    p.pixel_format = Some(PixelFormat::Rgba);
    p
}

fuzz_target!(|data: &[u8]| {
    let Ok(mut dec) = pgs::make_decoder(&params()) else {
        return;
    };
    let mut pos = 0usize;
    let mut packets = 0u32;
    while pos < data.len() && packets < 256 {
        let ctrl = data[pos];
        pos += 1;
        let do_reset = ctrl & 0x80 != 0;
        let want = (ctrl & 0x7f) as usize * 4; // chunk length up to 508 bytes
        let end = (pos + want).min(data.len());
        let chunk = &data[pos..end];
        pos = end;

        if do_reset {
            let _ = dec.reset();
        }
        let pkt = Packet::new(0, TimeBase::new(1, 90_000), chunk.to_vec()).with_pts(packets as i64);
        if dec.send_packet(&pkt).is_ok() {
            let mut guard = 0u32;
            while dec.receive_frame().is_ok() {
                guard += 1;
                if guard > 1024 {
                    panic!("PGS decoder produced an unbounded frame stream");
                }
            }
        }
        packets += 1;
    }
});
