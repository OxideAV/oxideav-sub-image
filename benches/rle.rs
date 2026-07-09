//! Criterion benchmarks for the bitmap-subtitle run-length hot loops.
//!
//! Round 403 (depth-mode benchmarks): oxideav-sub-image has reached the
//! per-format saturation point — the PGS / DVB / VobSub parsers are fuzz-
//! clean and spec-pinned — so per the workspace "saturated →
//! fuzz/bench/profile" memo this round wires up `criterion` benches over
//! the run-length coders that dominate decode/encode CPU. Future
//! optimisation rounds can A/B-test their changes against these.
//!
//! Everything is generated on the fly from a deterministic LCG — no
//! `docs/` fixtures or external files are read. The synthetic planes mimic
//! real subtitle content: a mostly-transparent (index 0) background broken
//! up by short glyph strokes and the occasional long horizontal fill, which
//! is exactly the run mix the RLE forms are tuned for.
//!
//!   - **pgs_encode / pgs_decode**: the PGS `encode_rle` / `decode_rle`
//!     scanline coders (0x00-escape runs, 14-bit lengths, end-of-line
//!     markers) on a 720×90 palette-index plane — a typical bottom-third
//!     Blu-ray caption band.
//!   - **dvb_encode_2bit / _4bit / _8bit**: the DVB pixel-code-string
//!     emitters (`emit_2bit_run` / `emit_4bit_run` and the byte-aligned
//!     8-bit path) over one 720-pixel scanline at each region depth,
//!     covering the run_length_3-10 / 12-27 / 29-284 form-selection ladder.
//!
//! Run with:
//!     cargo bench -p oxideav-sub-image --bench rle

use criterion::{criterion_group, criterion_main, Criterion, Throughput};

use oxideav_sub_image::dvbsub;
use oxideav_sub_image::pgs;

/// Tiny deterministic LCG so the benches are reproducible without pulling
/// in an RNG crate.
fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 33
}

/// Build a `width`×`height` palette-index plane with `max_index` distinct
/// colours: a mostly-zero (transparent) field sprinkled with short strokes
/// and occasional long runs — the workload the RLE forms are shaped for.
fn subtitle_plane(width: usize, height: usize, max_index: u8) -> Vec<u8> {
    let mut state: u64 = 0x51E6_2A3C_9F14_88D7;
    let mut plane = vec![0u8; width * height];
    for row in plane.chunks_mut(width) {
        let mut col = 0usize;
        while col < width {
            let roll = lcg(&mut state) % 100;
            let (colour, run) = if roll < 70 {
                // Background gap.
                (0u8, 3 + (lcg(&mut state) % 40) as usize)
            } else if roll < 95 {
                // A short glyph stroke.
                (
                    1 + (lcg(&mut state) % max_index as u64) as u8,
                    1 + (lcg(&mut state) % 6) as usize,
                )
            } else {
                // An occasional long uniform fill.
                (
                    (lcg(&mut state) % (max_index as u64 + 1)) as u8,
                    20 + (lcg(&mut state) % 120) as usize,
                )
            };
            let end = (col + run).min(width);
            for px in &mut row[col..end] {
                *px = colour;
            }
            col = end;
        }
    }
    plane
}

fn bench_pgs(c: &mut Criterion) {
    let (w, h) = (720usize, 90usize);
    let plane = subtitle_plane(w, h, 200);
    let encoded = pgs::encode_rle(&plane, w, h);

    let mut group = c.benchmark_group("pgs_rle");
    group.throughput(Throughput::Elements((w * h) as u64));
    group.bench_function("encode_720x90", |b| {
        b.iter(|| pgs::encode_rle(std::hint::black_box(&plane), w, h));
    });
    group.throughput(Throughput::Bytes(encoded.len() as u64));
    group.bench_function("decode_720x90", |b| {
        b.iter(|| pgs::decode_rle(std::hint::black_box(&encoded), w, h).unwrap());
    });
    group.finish();
}

fn bench_dvb(c: &mut Criterion) {
    let w = 720usize;
    let row2: Vec<u8> = subtitle_plane(w, 1, 3);
    let row4: Vec<u8> = subtitle_plane(w, 1, 15);
    let row8: Vec<u8> = subtitle_plane(w, 1, 255);

    let mut group = c.benchmark_group("dvb_pixel_string");
    group.throughput(Throughput::Elements(w as u64));
    group.bench_function("encode_2bit_row720", |b| {
        b.iter(|| dvbsub::encode_2bit_pixel_string(std::hint::black_box(&row2)).unwrap());
    });
    group.bench_function("encode_4bit_row720", |b| {
        b.iter(|| dvbsub::encode_4bit_pixel_string(std::hint::black_box(&row4)).unwrap());
    });
    group.bench_function("encode_8bit_row720", |b| {
        b.iter(|| dvbsub::encode_8bit_pixel_string(std::hint::black_box(&row8)));
    });
    group.finish();
}

criterion_group!(benches, bench_pgs, bench_dvb);
criterion_main!(benches);
