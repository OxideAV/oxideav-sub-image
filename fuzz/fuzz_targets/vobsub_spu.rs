#![no_main]

//! VobSub / DVD-SPU decode fuzz target.
//!
//! Feeds arbitrary bytes to `parse_and_decode_spu`, which parses the
//! Sub-Picture Unit header (`SPDSZ` total-size + `SP_DCSQTA` control-table
//! pointer), RLE-decodes the two interleaved pixel fields, and walks the
//! Display Control Sequence chain via self-relative `SP_NXT_DCSQ_SA`
//! offsets (SET_COLOR / SET_CONTR / SET_DAREA / SET_DSPXA / CHG_COLCON /
//! STA_DSP / STP_DSP / FSTA_DSP). The pointer chain is the headline
//! hazard: a `SP_NXT_DCSQ_SA` that points backward or to itself short of
//! the terminator is a classic decode-loop trap, and the
//! attacker-controlled `SP_DCSQTA` / display-area / pixel-address offsets
//! drive every slice into the buffer.
//!
//! Contract: **no-panic** and **termination** — the function must return a
//! `Result` rather than panicking, overflowing, indexing out of bounds,
//! OOM-aborting, or looping forever on a malicious DCSQ chain.

use libfuzzer_sys::fuzz_target;
use oxideav_sub_image::vobsub;

fuzz_target!(|data: &[u8]| {
    // Errors are expected and fine; we only care that it never crashes or
    // hangs. libfuzzer's own timeout backstops a runaway loop, but the
    // decoder is expected to bound its DCSQ traversal itself.
    let _ = vobsub::parse_and_decode_spu(data);
});
