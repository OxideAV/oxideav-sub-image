#![no_main]

//! VobSub `.idx` text-index parser fuzz target.
//!
//! Feeds arbitrary bytes (interpreted as a UTF-8 `.idx` text file) to
//! `parse_idx`, which scans `size:`, `palette:`, and `timestamp:` lines
//! and parses their hex / decimal / `WxH` / `HH:MM:SS:mmm` payloads. The
//! parser does its own numeric conversion on attacker-controlled text, so
//! a malformed dimension, an over-long palette list, or a junk timestamp
//! must produce an `Err`, never a panic.
//!
//! Contract: **no-panic** for any byte string that happens to be valid
//! UTF-8.

use libfuzzer_sys::fuzz_target;
use oxideav_sub_image::vobsub;

fuzz_target!(|data: &[u8]| {
    if let Ok(text) = std::str::from_utf8(data) {
        let _ = vobsub::parse_idx(text);
    }
});
