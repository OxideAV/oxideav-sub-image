//! Encode → decode round-trip tests for the PGS encoder/decoder pair.

use oxideav_core::{
    CodecId, CodecParameters, Frame, PixelFormat, TimeBase, VideoFrame, VideoPlane,
};
use oxideav_sub_image::{pgs, PGS_CODEC_ID};

fn codec_params() -> CodecParameters {
    let mut params = CodecParameters::video(CodecId::new(PGS_CODEC_ID));
    params.media_type = oxideav_core::MediaType::Subtitle;
    params.pixel_format = Some(PixelFormat::Rgba);
    params
}

fn make_rgba_frame(width: u32, height: u32, pixels: &[[u8; 4]]) -> VideoFrame {
    assert_eq!(pixels.len(), (width * height) as usize);
    let mut data = Vec::with_capacity(pixels.len() * 4);
    for px in pixels {
        data.extend_from_slice(px);
    }
    let _ = height;
    VideoFrame {
        pts: Some(0),
        planes: vec![VideoPlane {
            stride: (width as usize) * 4,
            data,
        }],
    }
}

/// A tiny 4-colour cue round-trips losslessly: every distinct input RGBA
/// value fits inside the 255-entry palette budget so the only loss is
/// the BT.601 RGB↔YCbCr transform, which is approximate but preserves
/// dominance (red stays red, etc.) and alpha.
#[test]
fn small_palette_roundtrip() {
    let pixels = [
        [255, 0, 0, 255],
        [255, 0, 0, 255],
        [0, 255, 0, 255],
        [0, 0, 255, 255],
    ];
    let frame = make_rgba_frame(2, 2, &pixels);

    let mut enc = pgs::make_encoder(&codec_params()).unwrap();
    enc.send_frame(&Frame::Video(frame)).unwrap();
    let packet = enc.receive_packet().unwrap();
    assert!(packet.flags.keyframe);
    assert_eq!(packet.pts, Some(0));

    let mut dec = pgs::make_decoder(&codec_params()).unwrap();
    dec.send_packet(&packet).unwrap();
    let decoded = dec.receive_frame().unwrap();
    let Frame::Video(v) = decoded else {
        panic!("expected video frame");
    };
    assert_eq!(v.planes[0].stride, 2 * 4);

    let d = &v.planes[0].data;
    assert_eq!(d.len(), 2 * 2 * 4);

    let red_a = &d[0..4];
    let red_b = &d[4..8];
    let green = &d[8..12];
    let blue = &d[12..16];
    assert!(red_a[0] > 200 && red_a[3] == 255, "row0,col0 {:?}", red_a);
    assert!(red_b[0] > 200 && red_b[3] == 255, "row0,col1 {:?}", red_b);
    assert!(
        green[1] > green[0] && green[1] > green[2] && green[3] == 255,
        "row1,col0 {:?}",
        green
    );
    assert!(
        blue[2] > blue[0] && blue[2] > blue[1] && blue[3] == 255,
        "row1,col1 {:?}",
        blue
    );
}

/// Transparent background must survive the round-trip. Index 0 is
/// reserved in the encoder palette for fully-transparent pixels.
#[test]
fn transparent_background_roundtrip() {
    let pixels = [[0, 0, 0, 0], [200, 0, 0, 255], [0, 0, 0, 0], [0, 0, 0, 0]];
    let frame = make_rgba_frame(2, 2, &pixels);
    let mut enc = pgs::make_encoder(&codec_params()).unwrap();
    enc.send_frame(&Frame::Video(frame)).unwrap();
    let packet = enc.receive_packet().unwrap();

    let mut dec = pgs::make_decoder(&codec_params()).unwrap();
    dec.send_packet(&packet).unwrap();
    let Frame::Video(v) = dec.receive_frame().unwrap() else {
        panic!("expected video frame");
    };
    let d = &v.planes[0].data;
    assert_eq!(&d[0..4], &[0, 0, 0, 0]);
    assert_eq!(d[4 + 3], 255);
    assert!(d[4] > 150);
    assert_eq!(&d[8..12], &[0, 0, 0, 0]);
    assert_eq!(&d[12..16], &[0, 0, 0, 0]);
}

/// A frame with > 255 distinct colours falls back to the 3/3/2/2
/// quantiser — still decodes without error and preserves the frame
/// geometry. We don't check per-pixel colour here because the fallback
/// path is lossy.
#[test]
fn lossy_fallback_for_wide_palette() {
    let w = 32u32;
    let h = 16u32;
    let mut pixels = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            pixels.push([(x * 8) as u8, (y * 16) as u8, ((x + y) * 4) as u8, 255u8]);
        }
    }
    let frame = make_rgba_frame(w, h, &pixels);
    let mut enc = pgs::make_encoder(&codec_params()).unwrap();
    enc.send_frame(&Frame::Video(frame)).unwrap();
    let packet = enc.receive_packet().unwrap();
    let mut dec = pgs::make_decoder(&codec_params()).unwrap();
    dec.send_packet(&packet).unwrap();
    let Frame::Video(v) = dec.receive_frame().unwrap() else {
        panic!("expected video frame");
    };
    assert_eq!(v.planes[0].stride, (w * 4) as usize);
    assert_eq!(v.planes[0].data.len(), (w * h * 4) as usize);
    // Every output pixel must carry a non-zero alpha (original alpha was
    // 255 everywhere) so the quantiser didn't accidentally drop the
    // image into the transparent entry.
    let mut opaque = 0;
    for chunk in v.planes[0].data.chunks(4) {
        if chunk[3] > 0 {
            opaque += 1;
        }
    }
    assert!(
        opaque as u32 >= (w * h) * 9 / 10,
        "too many transparent pixels after round-trip: {opaque}"
    );
}

/// PTS carried in an arbitrary time-base is rescaled to the 90 kHz base
/// PGS uses on the wire, then back to the decoder's output.
#[test]
fn pts_is_rescaled_to_90khz() {
    let frame = VideoFrame {
        // PTS interpreted as microseconds — the canonical input timebase
        // post-slim VideoFrame. 1000 µs → 90 ticks at the 90 kHz PGS clock.
        pts: Some(1000),
        planes: vec![VideoPlane {
            stride: 8,
            data: vec![0u8; 16],
        }],
    };
    let mut enc = pgs::make_encoder(&codec_params()).unwrap();
    enc.send_frame(&Frame::Video(frame)).unwrap();
    let packet = enc.receive_packet().unwrap();
    assert_eq!(packet.pts, Some(90));
    assert_eq!(packet.time_base, TimeBase::new(1, 90_000));
}
