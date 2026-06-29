#![no_main]

//! DVB-subtitle full-display-set decode fuzz target.
//!
//! Feeds arbitrary bytes as one PES-payload packet to the public DVB
//! subtitle decoder and drains every frame. The decoder strips the
//! optional `0x20 0x00` data-identifier prefix, walks the `0x0F`-synced
//! segments (page-composition / region-composition / CLUT-definition /
//! object-data / display-definition / disparity-signalling / end), runs
//! the epoch state machine, applies map-tables, parses character-coded
//! objects, honours the region fill flag + non-modifying-colour flag, and
//! decodes the 2/4/8-bit pixel-code strings — all of which read
//! attacker-controlled length / count / position / depth fields.
//!
//! Contract: **no-panic**. Every entry point must return a `Result`
//! rather than panicking, overflowing, indexing out of bounds, OOM-
//! aborting, or hanging.

use libfuzzer_sys::fuzz_target;
use oxideav_core::{CodecId, CodecParameters, MediaType, Packet, PixelFormat, TimeBase};
use oxideav_sub_image::{dvbsub, DVBSUB_CODEC_ID};

fn params() -> CodecParameters {
    let mut p = CodecParameters::video(CodecId::new(DVBSUB_CODEC_ID));
    p.media_type = MediaType::Subtitle;
    p.pixel_format = Some(PixelFormat::Rgba);
    p
}

fuzz_target!(|data: &[u8]| {
    let Ok(mut dec) = dvbsub::make_decoder(&params()) else {
        return;
    };
    let pkt = Packet::new(0, TimeBase::new(1, 90_000), data.to_vec()).with_pts(0);
    if dec.send_packet(&pkt).is_err() {
        return;
    }
    let mut guard = 0u32;
    while dec.receive_frame().is_ok() {
        guard += 1;
        if guard > 4096 {
            panic!("DVB subtitle decoder produced an unbounded frame stream");
        }
    }
});
