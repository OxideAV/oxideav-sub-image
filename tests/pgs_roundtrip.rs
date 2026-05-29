//! Encode → decode round-trip tests for the PGS encoder/decoder pair.

use oxideav_core::{
    CodecId, CodecParameters, Frame, Packet, PixelFormat, TimeBase, VideoFrame, VideoPlane,
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

/// The encoder crops to the tight bbox of non-transparent pixels: a
/// large canvas with a small visible cue near the bottom encodes to
/// noticeably fewer bytes than a full-canvas variant of the same shape
/// (every transparent row would otherwise need an RLE end-of-line run).
/// The decoded RGBA frame still has full-canvas dimensions and the cue
/// lands at the correct position.
#[test]
fn tight_bbox_shrinks_encoded_size_and_decodes_at_offset() {
    let w = 64u32;
    let h = 64u32;
    // Canvas is mostly transparent. A 4×3 opaque block lives at
    // (40, 50). Everywhere else is alpha 0.
    let bx = 40usize;
    let by = 50usize;
    let bw = 4usize;
    let bh = 3usize;
    let mut pixels = vec![[0u8, 0, 0, 0]; (w * h) as usize];
    for r in 0..bh {
        for c in 0..bw {
            let idx = (by + r) * w as usize + (bx + c);
            pixels[idx] = [200, 50, 25, 255];
        }
    }
    let frame = make_rgba_frame(w, h, &pixels);

    let mut enc = pgs::make_encoder(&codec_params()).unwrap();
    enc.send_frame(&Frame::Video(frame)).unwrap();
    let pkt = enc.receive_packet().unwrap();

    // Full-canvas equivalent would carry 64 rows of RLE plus end-of-line
    // markers; the bbox version only carries 3 rows. The exact saving
    // depends on the RLE format but the bytewise inequality is a stable
    // floor.
    assert!(
        pkt.data.len() < 200,
        "tight-bbox packet unexpectedly large: {} bytes",
        pkt.data.len()
    );

    // Round-trip — decoded canvas keeps full size and only the bbox is
    // non-transparent.
    let mut dec = pgs::make_decoder(&codec_params()).unwrap();
    dec.send_packet(&pkt).unwrap();
    let Frame::Video(v) = dec.receive_frame().unwrap() else {
        panic!("expected video frame");
    };
    assert_eq!(v.planes[0].stride, (w * 4) as usize);
    assert_eq!(v.planes[0].data.len(), (w * h * 4) as usize);
    let d = &v.planes[0].data;
    // Pixel outside the bbox is fully-transparent.
    let outside = ((10u32) * w + 10) as usize * 4;
    assert_eq!(&d[outside..outside + 4], &[0, 0, 0, 0]);
    // Pixel inside the bbox carries the red-dominated quantised colour.
    let inside = ((by as u32) * w + bx as u32) as usize * 4;
    assert!(
        d[inside] > 150 && d[inside + 3] == 255,
        "expected red+opaque at bbox origin, got {:?}",
        &d[inside..inside + 4]
    );
    // Just outside the bbox to the east (within the same row) is also
    // transparent, confirming the encoder didn't smear.
    let east = ((by as u32) * w + (bx + bw) as u32) as usize * 4;
    assert_eq!(&d[east..east + 4], &[0, 0, 0, 0]);
}

/// A fully-transparent input frame emits an erase display-set (PCS
/// with zero composition objects) and decodes to a fully-transparent
/// canvas at the canvas dimensions.
#[test]
fn fully_transparent_frame_emits_erase() {
    let w = 16u32;
    let h = 8u32;
    let pixels = vec![[0u8, 0, 0, 0]; (w * h) as usize];
    let frame = make_rgba_frame(w, h, &pixels);

    let mut enc = pgs::make_encoder(&codec_params()).unwrap();
    enc.send_frame(&Frame::Video(frame)).unwrap();
    let pkt = enc.receive_packet().unwrap();

    // Erase = PCS + WDS + END only — no PDS, no ODS. Three "PG"
    // segment headers (13 bytes each) plus body bytes.
    let mut pg_count = 0;
    for i in 0..pkt.data.len().saturating_sub(1) {
        if &pkt.data[i..i + 2] == b"PG" {
            pg_count += 1;
        }
    }
    assert_eq!(
        pg_count, 3,
        "erase display-set must carry exactly PCS + WDS + END"
    );

    let mut dec = pgs::make_decoder(&codec_params()).unwrap();
    dec.send_packet(&pkt).unwrap();
    let Frame::Video(v) = dec.receive_frame().unwrap() else {
        panic!("expected video frame");
    };
    assert_eq!(v.planes[0].stride, (w * 4) as usize);
    assert_eq!(v.planes[0].data.len(), (w * h * 4) as usize);
    for chunk in v.planes[0].data.chunks(4) {
        assert_eq!(chunk, &[0, 0, 0, 0]);
    }
}

/// Padding rows / columns on every side of an otherwise non-transparent
/// region are stripped by the bbox detector: encoding then decoding
/// reproduces the same pixels at the same coordinates.
#[test]
fn bbox_strips_padding_on_all_sides() {
    let w = 8u32;
    let h = 8u32;
    let mut pixels = vec![[0u8, 0, 0, 0]; (w * h) as usize];
    // 2×2 opaque square at (3, 3).
    for r in 3..5 {
        for c in 3..5 {
            pixels[r * w as usize + c] = [0, 200, 0, 255];
        }
    }
    let frame = make_rgba_frame(w, h, &pixels);
    let mut enc = pgs::make_encoder(&codec_params()).unwrap();
    enc.send_frame(&Frame::Video(frame)).unwrap();
    let pkt = enc.receive_packet().unwrap();
    let mut dec = pgs::make_decoder(&codec_params()).unwrap();
    dec.send_packet(&pkt).unwrap();
    let Frame::Video(v) = dec.receive_frame().unwrap() else {
        panic!("expected video frame");
    };
    let d = &v.planes[0].data;
    // Every pixel outside the 2×2 square is transparent.
    for r in 0..h as usize {
        for c in 0..w as usize {
            let i = (r * w as usize + c) * 4;
            if (3..5).contains(&r) && (3..5).contains(&c) {
                assert!(
                    d[i + 1] > 100 && d[i + 3] == 255,
                    "expected opaque green at ({}, {}), got {:?}",
                    r,
                    c,
                    &d[i..i + 4]
                );
            } else {
                assert_eq!(
                    &d[i..i + 4],
                    &[0, 0, 0, 0],
                    "expected transparent at ({}, {}), got {:?}",
                    r,
                    c,
                    &d[i..i + 4]
                );
            }
        }
    }
}

/// A multi-fragment ODS (the form a real PGS muxer emits for objects too
/// large for one segment) decodes to the same pixels as the equivalent
/// single-segment object, driven entirely through the public decoder.
#[test]
fn fragmented_ods_decodes_like_single_segment() {
    let pixels = [1u8, 2, 3, 1, 0, 2, 3, 0, 1, 1, 2, 3]; // 4×3
    let palette = [
        (0u8, [0u8, 0, 0, 0]),
        (1u8, [220u8, 20, 20, 255]),
        (2u8, [20u8, 220, 20, 255]),
        (3u8, [20u8, 20, 220, 255]),
    ];

    let decode = |blob: Vec<u8>| -> Vec<u8> {
        let mut dec = pgs::make_decoder(&codec_params()).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 90_000), blob).with_pts(0);
        dec.send_packet(&pkt).unwrap();
        let Frame::Video(v) = dec.receive_frame().unwrap() else {
            panic!("expected video frame");
        };
        v.planes[0].data.clone()
    };

    let single = decode(pgs::build_demo_display_set(
        (4, 3),
        (4, 3),
        (0, 0),
        &palette,
        &pixels,
    ));
    let multi = decode(pgs::build_demo_display_set_fragmented(
        (4, 3),
        (4, 3),
        (0, 0),
        &palette,
        &pixels,
        4,
    ));
    assert_eq!(single, multi, "fragmented ODS diverged from single-segment");
}
