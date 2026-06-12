//! Encode → decode round-trip tests for the DVB subtitle encoder /
//! decoder pair, driven entirely through the public factory APIs.

use oxideav_core::{
    CodecId, CodecParameters, Frame, PixelFormat, TimeBase, VideoFrame, VideoPlane,
};
use oxideav_sub_image::{dvbsub, DVBSUB_CODEC_ID};

fn codec_params() -> CodecParameters {
    let mut params = CodecParameters::video(CodecId::new(DVBSUB_CODEC_ID));
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
    VideoFrame {
        pts: Some(0),
        planes: vec![VideoPlane {
            stride: (width as usize) * 4,
            data,
        }],
    }
}

fn encode_one(frame: VideoFrame) -> oxideav_core::Packet {
    let mut enc = dvbsub::make_encoder(&codec_params()).unwrap();
    enc.send_frame(&Frame::Video(frame)).unwrap();
    enc.receive_packet().unwrap()
}

fn decode_one(packet: &oxideav_core::Packet) -> VideoFrame {
    let mut dec = dvbsub::make_decoder(&codec_params()).unwrap();
    dec.send_packet(packet).unwrap();
    let Frame::Video(v) = dec.receive_frame().unwrap() else {
        panic!("expected video frame");
    };
    v
}

/// Walk the encoded PES payload (skipping the `0x20 0x00` prefix) and
/// collect `(segment_type, body)` pairs via the public segment reader.
fn segments(packet: &oxideav_core::Packet) -> Vec<(u8, Vec<u8>)> {
    assert_eq!(
        &packet.data[..2],
        &[0x20, 0x00],
        "payload must start with the data_identifier / stream_id prefix"
    );
    let buf = &packet.data[2..];
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < buf.len() {
        let (seg, next) = dvbsub::read_segment(buf, pos).unwrap();
        out.push((seg.seg_type, seg.body));
        pos = next;
    }
    out
}

/// First data_type byte of the object-data top-field block — the
/// pixel-code-string flavour the encoder chose (body layout: object_id
/// (2) + flags (1) + top_len (2) + bottom_len (2) + top block).
fn ods_data_type(packet: &oxideav_core::Packet) -> u8 {
    let segs = segments(packet);
    let (_, body) = segs
        .iter()
        .find(|(t, _)| *t == dvbsub::SEG_OBJECT_DATA)
        .expect("display set carries no object-data segment");
    body[7]
}

/// Greys round-trip bit-exactly (the BT.601 chroma terms vanish), and a
/// 4-entry palette (transparent + 3 greys) selects the 2-bit
/// pixel-code-string path.
#[test]
fn grey_2bit_roundtrip_is_bit_exact() {
    let g1 = [60u8, 60, 60, 255];
    let g2 = [128u8, 128, 128, 255];
    let g3 = [220u8, 220, 220, 255];
    let tr = [0u8, 0, 0, 0];
    let pixels = [
        g1, g2, g3, tr, //
        g3, g3, g1, g2, //
        tr, g2, g2, g1, //
    ];
    let frame = make_rgba_frame(4, 3, &pixels);
    let pkt = encode_one(frame);
    assert!(pkt.flags.keyframe);
    assert_eq!(ods_data_type(&pkt), dvbsub::DATA_TYPE_2BIT);

    let v = decode_one(&pkt);
    assert_eq!(v.planes[0].stride, 4 * 4);
    assert_eq!(v.planes[0].data.len(), 4 * 3 * 4);
    for (i, want) in pixels.iter().enumerate() {
        let got = &v.planes[0].data[i * 4..i * 4 + 4];
        assert_eq!(got, want, "pixel {i} did not roundtrip");
    }
}

/// A 5..16-entry palette selects the 4-bit path; grey payloads stay
/// bit-exact end to end.
#[test]
fn grey_4bit_roundtrip_is_bit_exact() {
    let w = 10u32;
    let h = 2u32;
    // 10 distinct greys (+ transparent) → 11 palette entries → 4-bit.
    let mut pixels = Vec::new();
    for row in 0..h {
        for col in 0..w {
            if row == 1 && col == 0 {
                pixels.push([0u8, 0, 0, 0]);
            } else {
                let v = (20 + col * 22) as u8;
                pixels.push([v, v, v, 255]);
            }
        }
    }
    let frame = make_rgba_frame(w, h, &pixels);
    let pkt = encode_one(frame);
    assert_eq!(ods_data_type(&pkt), dvbsub::DATA_TYPE_4BIT);

    let v = decode_one(&pkt);
    for (i, want) in pixels.iter().enumerate() {
        let got = &v.planes[0].data[i * 4..i * 4 + 4];
        assert_eq!(got, want, "pixel {i} did not roundtrip");
    }
}

/// More than 16 palette entries selects the 8-bit path.
#[test]
fn grey_8bit_roundtrip_is_bit_exact() {
    let w = 32u32;
    let h = 2u32;
    let mut pixels = Vec::new();
    for row in 0..h {
        for col in 0..w {
            let v = (col * 8 + row * 4) as u8;
            pixels.push([v, v, v, 255]);
        }
    }
    let frame = make_rgba_frame(w, h, &pixels);
    let pkt = encode_one(frame);
    assert_eq!(ods_data_type(&pkt), dvbsub::DATA_TYPE_8BIT);

    let v = decode_one(&pkt);
    for (i, want) in pixels.iter().enumerate() {
        let got = &v.planes[0].data[i * 4..i * 4 + 4];
        assert_eq!(got, want, "pixel {i} did not roundtrip");
    }
}

/// Colour payloads survive the BT.601 RGB↔YCbCr trip with channel
/// dominance preserved and alpha exact.
#[test]
fn colour_roundtrip_preserves_dominance_and_alpha() {
    let pixels = [
        [255u8, 0, 0, 255],
        [0, 255, 0, 255],
        [0, 0, 255, 255],
        [200, 50, 25, 128],
    ];
    let frame = make_rgba_frame(2, 2, &pixels);
    let v = decode_one(&encode_one(frame));
    let d = &v.planes[0].data;
    assert!(
        d[0] > d[1] && d[0] > d[2],
        "red lost dominance: {:?}",
        &d[0..4]
    );
    assert!(
        d[5] > d[4] && d[5] > d[6],
        "green lost dominance: {:?}",
        &d[4..8]
    );
    assert!(
        d[10] > d[8] && d[10] > d[9],
        "blue lost dominance: {:?}",
        &d[8..12]
    );
    for (i, want) in pixels.iter().enumerate() {
        assert_eq!(d[i * 4 + 3], want[3], "alpha must roundtrip exactly");
        for c in 0..3 {
            assert!(
                (d[i * 4 + c] as i32 - want[c] as i32).abs() <= 2,
                "pixel {i} channel {c} drifted: {:?} vs {want:?}",
                &d[i * 4..i * 4 + 4]
            );
        }
    }
}

/// The encoder crops to the tight bbox of non-transparent pixels and
/// places the region through the page composition; the decoded canvas
/// keeps full frame geometry with the cue at the original offset.
#[test]
fn bbox_crop_decodes_at_original_offset() {
    let w = 64u32;
    let h = 48u32;
    let (bx, by, bw, bh) = (40usize, 30usize, 5usize, 4usize);
    let grey = [190u8, 190, 190, 255];
    let mut pixels = vec![[0u8, 0, 0, 0]; (w * h) as usize];
    for r in 0..bh {
        for c in 0..bw {
            pixels[(by + r) * w as usize + (bx + c)] = grey;
        }
    }
    let frame = make_rgba_frame(w, h, &pixels);
    let pkt = encode_one(frame);
    // The cropped object is 5×4 — the whole display set stays small.
    assert!(
        pkt.data.len() < 200,
        "bbox-cropped packet unexpectedly large: {} bytes",
        pkt.data.len()
    );
    let v = decode_one(&pkt);
    assert_eq!(v.planes[0].stride, (w * 4) as usize);
    assert_eq!(v.planes[0].data.len(), (w * h * 4) as usize);
    let d = &v.planes[0].data;
    for r in 0..h as usize {
        for c in 0..w as usize {
            let i = (r * w as usize + c) * 4;
            let inside = (by..by + bh).contains(&r) && (bx..bx + bw).contains(&c);
            if inside {
                assert_eq!(&d[i..i + 4], &grey, "expected grey at ({c}, {r})");
            } else {
                assert_eq!(&d[i..i + 4], &[0, 0, 0, 0], "expected clear at ({c}, {r})");
            }
        }
    }
}

/// A fully-transparent frame encodes an erase display set: a page
/// composition referencing no regions, no region/CLUT/object segments,
/// and a fully-transparent decoded canvas.
#[test]
fn fully_transparent_frame_emits_erase_set() {
    let w = 16u32;
    let h = 8u32;
    let pixels = vec![[0u8, 0, 0, 0]; (w * h) as usize];
    let pkt = encode_one(make_rgba_frame(w, h, &pixels));
    let segs = segments(&pkt);
    let types: Vec<u8> = segs.iter().map(|(t, _)| *t).collect();
    assert_eq!(
        types,
        vec![
            dvbsub::SEG_DISPLAY_DEFINITION,
            dvbsub::SEG_PAGE_COMPOSITION,
            dvbsub::SEG_END_OF_DISPLAY_SET,
        ],
        "erase set must carry exactly DDS + PCS + END"
    );
    let v = decode_one(&pkt);
    assert_eq!(v.planes[0].data.len(), (w * h * 4) as usize);
    for chunk in v.planes[0].data.chunks(4) {
        assert_eq!(chunk, &[0, 0, 0, 0]);
    }
}

/// Frame pts (microseconds) is rescaled to the 90 kHz clock the packet
/// declares.
#[test]
fn pts_is_rescaled_to_90khz() {
    let frame = VideoFrame {
        pts: Some(1000), // 1000 µs → 90 ticks at 90 kHz
        planes: vec![VideoPlane {
            stride: 8,
            data: vec![0u8; 16],
        }],
    };
    let pkt = encode_one(frame);
    assert_eq!(pkt.pts, Some(90));
    assert_eq!(pkt.time_base, TimeBase::new(1, 90_000));
}

/// More than 255 distinct colours falls back to the 3-3-2-2 channel
/// reduction: the stream still decodes, geometry is preserved, and the
/// originally-opaque pixels stay opaque.
#[test]
fn wide_palette_falls_back_to_lossy_quantisation() {
    let w = 32u32;
    let h = 16u32;
    let mut pixels = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            pixels.push([(x * 8) as u8, (y * 16) as u8, ((x + y) * 4) as u8, 255u8]);
        }
    }
    let pkt = encode_one(make_rgba_frame(w, h, &pixels));
    assert_eq!(ods_data_type(&pkt), dvbsub::DATA_TYPE_8BIT);
    let v = decode_one(&pkt);
    assert_eq!(v.planes[0].stride, (w * 4) as usize);
    assert_eq!(v.planes[0].data.len(), (w * h * 4) as usize);
    for (i, chunk) in v.planes[0].data.chunks(4).enumerate() {
        assert!(chunk[3] > 0, "pixel {i} lost its opacity in the fallback");
    }
}

/// Two consecutive sends produce two independent display sets that each
/// decode standalone (DVB display sets rebuild state per PES payload).
#[test]
fn consecutive_frames_decode_independently() {
    let a = [200u8, 200, 200, 255];
    let b = [60u8, 60, 60, 255];
    let mut enc = dvbsub::make_encoder(&codec_params()).unwrap();
    enc.send_frame(&Frame::Video(make_rgba_frame(2, 1, &[a, a])))
        .unwrap();
    enc.send_frame(&Frame::Video(make_rgba_frame(2, 1, &[b, b])))
        .unwrap();
    let p1 = enc.receive_packet().unwrap();
    let p2 = enc.receive_packet().unwrap();
    let v1 = decode_one(&p1);
    let v2 = decode_one(&p2);
    assert_eq!(&v1.planes[0].data[0..4], &a);
    assert_eq!(&v2.planes[0].data[0..4], &b);
}
