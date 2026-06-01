# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- VobSub decoder now length-skips the SP_DCSQ CHG_COLCON command
  (opcode `0x07`) instead of hard-erroring on it. The command's
  documented self-delimiting form (one command byte + a 2-byte
  total-parameter-size word that includes the size word itself + a
  variable-length LN_CTLI / PX_CTLI payload terminated by the
  `0F FF FF FF` LN_CTLI sentinel) is consumed in lock-step with the
  rest of the control sequence: the mid-display palette / contrast
  mutations the command requests are not applied to the rendered
  bitmap, but SPUs that carry the command now decode their base
  palette / alpha / coords / RLE successfully where previously every
  control sequence containing a CHG_COLCON returned
  `InvalidData("vobsub SPU: unknown command 0x07")`. New
  `Spu::saw_chg_colcon` flag surfaces the fact that a stream asked
  for the mutations so callers / tests can observe the gap.
- Five new unit tests for the CHG_COLCON path: full SPU round-trip
  through a CHG_COLCON-bearing control sequence asserting that the
  decoded pixels match the CHG_COLCON-free baseline and the flag is
  set; CHG_COLCON with a `size = 2` (zero-payload) parameter block
  tolerated as a valid self-delimiting form; explicit error paths
  for size-word truncation, payload truncation, and a size word
  below the minimum legal value of 2.

## [0.0.7](https://github.com/OxideAV/oxideav-sub-image/compare/v0.0.6...v0.0.7) - 2026-05-29

### Other

- RLE property+negative sweep + scrub two attributive ffmpeg refs
- pgs encoder: tight-bbox crop + erase display-set for transparent frames
- harden blit_indexed against offset overflow + add property sweeps
- Porter–Duff source-over blit for overlapping subtitle objects
- re-document the subtitle-stream-emits-video-frames convention

### Added

- PGS multi-fragment ODS reassembly is now exercised by tests. A real
  PGS muxer splits any object whose data overruns the per-segment size
  limit across several `object_data` segments using the
  first/continuation/last sequence-flag bits, and the decoder must
  concatenate the fragments before reading the object_data_length /
  width / height header or the RLE. The single-segment demo builder
  never produced that shape, so the `parse_ods_into` reassembly branch
  was untested. New `build_demo_display_set_fragmented` test helper
  (sharing one `demo_rle` encoder with the single-segment builder so the
  two can't drift) splits the object payload across N segments at even
  byte boundaries — including boundaries inside the 7-byte width/height
  header. Tests assert: a 4×3 object fragmented into 2..=7 segments
  decodes byte-identically to the single-segment form; a 1×1 object cut
  into 32 one-byte fragments (first boundary inside object_data_length)
  still reassembles; a first-without-last (incomplete) object is dropped
  rather than rendered as a partial bitmap; plus an integration-level
  round-trip through the public decoder confirming the helper is reachable.
- PGS RLE codec property sweep: 1500 randomised encode→decode round-trips
  across varied widths (1..=31), heights (1..=17) and palette sizes
  (1..=12) confirm `decode_rle ∘ encode_rle == identity`; targeted sweeps
  for long uniform runs (widths 64/100/256/600/1024 over colour 0 + two
  non-zero colours), alternating short runs, 1×N single-column bitmaps,
  and a hand-crafted single-row that hits every encoder branch in
  sequence (1 literal → 3 singletons → short colour run → 14-bit colour
  run → 14-bit zero run). Also: size-shrink assertions on uniform
  bitmaps to keep the long-run encoder's optimality honest.
- PGS RLE decoder malformed-input sweep: targeted negative tests for
  truncated escape, truncated 14-bit-length, truncated short-colour-run,
  truncated 14-bit-colour-run; overlong-run row clamping; extra
  end-of-line past height; literal pixel past the final end-of-line; and
  a 400-iteration pseudo-random garbage sweep that asserts the decoder
  cannot be made to panic on arbitrary byte streams.
- `encode_rle` pre-sizes its output buffer to the worst-case
  singleton-plus-EOL bound, removing growth-churn allocations on typical
  subtitle-text bitmaps.

### Changed

- README + `pgs.rs` prose: rewrite two pre-existing decorative
  implementation-attribution phrases ("matching ffmpeg's convention" in
  the README and "ignore like ffmpeg does" in the segment-walker) into
  spec/behaviour-anchored prose. The algorithm description carries the
  load; the implementation attribution carries none.

- PGS encoder now crops each input frame to the tight bounding box of
  its non-transparent pixels before quantisation and emits a single
  composition object positioned at the bbox's `(x, y)`. The WDS window
  shrinks to match the object footprint. Fully-transparent input frames
  emit an erase display-set (PCS with zero composition objects + empty
  WDS) instead of paying the cost of a fully-transparent ODS RLE. For a
  100×40 cue on a 1920×1080 canvas the packet shrinks from ~8.4 KB to
  ~3.2 KB (~62%); for a wide subtitle text band the saving is closer to
  9%. The decoder side already handled arbitrary object dimensions and
  positions via `composite::blit_indexed`, so round-trips remain
  bit-identical for visible pixels.
- Three new round-trip tests covering the bbox path: a tight-bbox
  round-trip asserting both the smaller packet size and the correct
  pixel position; a fully-transparent-input → erase round-trip
  asserting PCS + WDS + END (no PDS / ODS); and an
  all-sides-padding-stripping round-trip exhaustively comparing every
  decoded pixel.
- `composite` module with a Porter–Duff source-over (`over`) primitive
  and an alpha-aware `blit_indexed` helper. Overlapping subtitle objects
  whose topmost pixel is partially transparent now blend over earlier
  canvas content instead of hard-overwriting it. PGS and DVB-subtitle
  render paths route their blits through the shared compositor.
- Property-style sweep tests for the compositor: `over` short-circuit
  identities over a swept destination-alpha space, output-alpha
  monotonicity (`Ao >= Ad`) across 200k random pixel pairs, `blit_indexed`
  agreeing bit-for-bit with an independent reference blit over random
  placements, split-tile vs. whole-tile blit equivalence, zero-size
  region no-ops, and pathological `usize::MAX`-class placement offsets.

### Fixed

- `blit_indexed` could panic (debug overflow check) or silently write to
  the wrong canvas pixel (release wrap) when a placement offset
  (`base_x`/`base_y`) near `usize::MAX` was supplied: the unchecked
  `base_x + col_idx` add wrapped to a small in-range value that slipped
  past the `dx >= width` clip. The destination coordinate is now computed
  with a checked add, so an overflowing offset clips off-canvas exactly
  like an out-of-range one. A `debug_assert` documents the
  `width * height * 4` canvas-length contract and a release-mode length
  guard skips rather than panics on an undersized canvas.

## [0.0.6](https://github.com/OxideAV/oxideav-sub-image/compare/v0.0.5...v0.0.6) - 2026-05-06

### Other

- drop stale REGISTRARS / with_all_features intra-doc links
- drop dead `linkme` dep
- auto-register via oxideav_core::register! macro (linkme distributed slice)
- unify entry point on register(&mut RuntimeContext) ([#502](https://github.com/OxideAV/oxideav-sub-image/pull/502))

## [0.0.5](https://github.com/OxideAV/oxideav-sub-image/compare/v0.0.4...v0.0.5) - 2026-05-03

### Other

- replace never-match regex with semver_check = false
- migrate to centralized OxideAV/.github reusable workflows
- adopt slim VideoFrame shape
- pin release-plz to patch-only bumps

## [0.0.4](https://github.com/OxideAV/oxideav-sub-image/compare/v0.0.3...v0.0.4) - 2026-04-25

### Other

- drop oxideav-codec/oxideav-container shims, import from oxideav-core
- bump oxideav-container dep to "0.1"
- drop Cargo.lock — this crate is a library
- bump oxideav-core / oxideav-codec dep examples to "0.1"
- bump to oxideav-core 0.1.1 + codec 0.1.1
- migrate register() to CodecInfo builder
- bump oxideav-core + oxideav-codec deps to "0.1"
- thread &dyn CodecResolver through open()
