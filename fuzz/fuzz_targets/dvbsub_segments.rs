#![no_main]

//! DVB-subtitle segment-reader fuzz target.
//!
//! Walks `read_segment` to exhaustion over arbitrary bytes so the
//! `0x0F` sync-byte / `segment_type` / `page_id` / `segment_length`
//! framing maths is fuzzed independently of the per-segment body
//! interpretation. A reader that mis-handles a truncated length field, a
//! zero-length segment, or a length that runs off the end of the buffer
//! is the classic place an off-by-one becomes an out-of-bounds slice.
//!
//! Contract: **no-panic**, and forward progress — each successful
//! `read_segment` must advance the cursor strictly, so the walk always
//! terminates.

use libfuzzer_sys::fuzz_target;
use oxideav_sub_image::dvbsub;

fuzz_target!(|data: &[u8]| {
    let mut pos = 0usize;
    let mut iters = 0u32;
    while pos < data.len() {
        match dvbsub::read_segment(data, pos) {
            Ok((_seg, next)) => {
                assert!(
                    next > pos,
                    "read_segment failed to advance the cursor (pos={pos}, next={next})"
                );
                pos = next;
            }
            Err(_) => break,
        }
        iters += 1;
        if iters > 100_000 {
            panic!("read_segment walk did not terminate");
        }
    }
});
