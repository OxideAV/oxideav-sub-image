#![no_main]

//! PGS run-length-decoder fuzz target.
//!
//! Narrows the input to "fuzzer-controlled width/height + arbitrary RLE
//! payload" so libfuzzer spends its whole iteration budget on the
//! per-scanline run-length state machine (the four code-word branches —
//! single-pixel literal, short/long preferred-colour run, short/long
//! arbitrary-colour run — plus the all-zeros end-of-line resync) rather
//! than re-discovering valid PG segment framing.
//!
//! Contract: **no-panic**. `decode_rle` must always return a `Result`;
//! a truncated escape, an over-long run, too many lines, or random
//! garbage must never panic, overflow, index out of bounds, or hang.

use libfuzzer_sys::fuzz_target;
use oxideav_sub_image::pgs;

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }
    // First four bytes seed the geometry; the rest is the RLE payload.
    // Cap dimensions so a valid decode allocates at most ~64 KiB.
    let width = (u16::from_le_bytes([data[0], data[1]]) % 256) as usize;
    let height = (u16::from_le_bytes([data[2], data[3]]) % 256) as usize;
    let rle = &data[4..];
    // Errors are expected and fine; we only care that it never crashes.
    let _ = pgs::decode_rle(rle, width, height);
});
