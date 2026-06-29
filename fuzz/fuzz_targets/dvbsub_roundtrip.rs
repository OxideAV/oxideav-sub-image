#![no_main]

//! DVB-subtitle encoder-roundtrip fuzz target.
//!
//! Drives the DVB subtitle encoder with a fuzzer-controlled RGBA frame,
//! then decodes the emitted display-set PES payload back through the
//! public decoder. Exercises the CLUT quantiser (≤4 → 2-bit, ≤16 →
//! 4-bit, else 8-bit, with the 3-3-2-2 wide-palette fallback), the three
//! run-length pixel-code-string encoders, the bbox crop + region
//! placement, and the fully-transparent erase-set path — and confirms the
//! bytes it produces decode without panicking.
//!
//! Contract: **no-panic** on both halves, and the decoded canvas keeps
//! the source geometry. Colour is lossy (BT.601 + quantisation) so
//! per-pixel equality is not asserted.

use libfuzzer_sys::fuzz_target;
use oxideav_core::{
    CodecId, CodecParameters, Frame, MediaType, PixelFormat, VideoFrame, VideoPlane,
};
use oxideav_sub_image::{dvbsub, DVBSUB_CODEC_ID};

fn params() -> CodecParameters {
    let mut p = CodecParameters::video(CodecId::new(DVBSUB_CODEC_ID));
    p.media_type = MediaType::Subtitle;
    p.pixel_format = Some(PixelFormat::Rgba);
    p
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }
    let width = (data[0] % 64 + 1) as u32;
    let height = (data[1] % 64 + 1) as u32;
    let need = (width * height * 4) as usize;
    let body = &data[2..];
    if body.is_empty() {
        return;
    }
    let mut plane = Vec::with_capacity(need);
    for i in 0..need {
        plane.push(body[i % body.len()]);
    }
    let frame = VideoFrame {
        pts: Some(0),
        planes: vec![VideoPlane {
            stride: (width * 4) as usize,
            data: plane,
        }],
    };

    let Ok(mut enc) = dvbsub::make_encoder(&params()) else {
        return;
    };
    if enc.send_frame(&Frame::Video(frame)).is_err() {
        return;
    }
    let Ok(pkt) = enc.receive_packet() else {
        return;
    };

    let Ok(mut dec) = dvbsub::make_decoder(&params()) else {
        return;
    };
    if dec.send_packet(&pkt).is_err() {
        return;
    }
    if let Ok(Frame::Video(v)) = dec.receive_frame() {
        assert_eq!(
            v.planes[0].data.len(),
            need,
            "decoded DVB canvas geometry diverged from the encoded frame"
        );
    }
});
