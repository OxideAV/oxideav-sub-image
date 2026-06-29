#![no_main]

//! PGS full-display-set decode fuzz target.
//!
//! Feeds arbitrary bytes as a single `.sup`-style packet payload to the
//! public PGS decoder and drains every frame it produces. The decoder
//! walks the 13-byte PG segment headers (PCS / WDS / PDS / ODS / END),
//! runs the epoch state machine, reassembles fragmented ODS, applies the
//! cropping rectangle + window clip, and RLE-decodes the object bitmaps —
//! every step of which reads attacker-controlled length / count / offset
//! fields.
//!
//! Contract: **no-panic**. `make_decoder`, `send_packet`, and
//! `receive_frame` must always return a `Result` rather than panicking,
//! integer-overflowing, indexing out of bounds, OOM-aborting, or looping
//! forever. Errors are fine; crashes are not.

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
    let pkt = Packet::new(0, TimeBase::new(1, 90_000), data.to_vec()).with_pts(0);
    if dec.send_packet(&pkt).is_err() {
        return;
    }
    // Drain frames; the decoder signals "no more" with an Err.
    let mut guard = 0u32;
    while dec.receive_frame().is_ok() {
        guard += 1;
        if guard > 4096 {
            // A well-behaved decoder yields a bounded number of frames per
            // packet; a runaway count is itself a defect worth surfacing.
            panic!("PGS decoder produced an unbounded frame stream");
        }
    }
});
