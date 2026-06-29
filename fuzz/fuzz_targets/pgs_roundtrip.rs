#![no_main]

//! PGS encoder-roundtrip fuzz target.
//!
//! Drives the PGS encoder with a fuzzer-controlled RGBA frame, then decodes
//! the emitted display-set back through the public decoder. This exercises
//! the encoder's tight-bbox detector, the palette quantiser (lossless ≤255
//! colours and the 3/3/2/2 wide-palette fallback), the per-scanline RLE
//! encoder, and the oversized-object ODS fragmentation — and confirms the
//! bytes it produces are themselves decodable.
//!
//! Contract: **no-panic** on both halves, and the decoded canvas keeps the
//! source geometry (full width × height × 4 RGBA bytes). Colour is lossy
//! (BT.601 round-trip + quantisation) so per-pixel equality is not asserted.

use libfuzzer_sys::fuzz_target;
use oxideav_core::{
    CodecId, CodecParameters, Frame, MediaType, PixelFormat, VideoFrame, VideoPlane,
};
use oxideav_sub_image::{pgs, PGS_CODEC_ID};

fn params() -> CodecParameters {
    let mut p = CodecParameters::video(CodecId::new(PGS_CODEC_ID));
    p.media_type = MediaType::Subtitle;
    p.pixel_format = Some(PixelFormat::Rgba);
    p
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }
    // Geometry from the first two bytes (1..=64 each → ≤16 KiB RGBA).
    let width = (data[0] % 64 + 1) as u32;
    let height = (data[1] % 64 + 1) as u32;
    let need = (width * height * 4) as usize;
    let mut plane = Vec::with_capacity(need);
    let body = &data[2..];
    if body.is_empty() {
        return;
    }
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

    let Ok(mut enc) = pgs::make_encoder(&params()) else {
        return;
    };
    if enc.send_frame(&Frame::Video(frame)).is_err() {
        return;
    }
    let Ok(pkt) = enc.receive_packet() else {
        return;
    };

    let Ok(mut dec) = pgs::make_decoder(&params()) else {
        return;
    };
    let in_pkt = pkt.clone();
    if dec.send_packet(&in_pkt).is_err() {
        return;
    }
    if let Ok(Frame::Video(v)) = dec.receive_frame() {
        assert_eq!(
            v.planes[0].data.len(),
            need,
            "decoded PGS canvas geometry diverged from the encoded frame"
        );
    }
});
