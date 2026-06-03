# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- PGS Composition Segments carrying `object_cropped_flag = 1` now
  **apply** the 8-byte cropping rectangle that trails the object entry,
  instead of dropping the rectangle bytes on the floor and compositing
  the full bitmap. `CompositionObject` gains a `crop: Option<CropRect>`
  field carrying the parsed `(x, y, w, h)` (four big-endian `u16`s in
  source-object space — the same rectangle layout used by the WDS window
  record elsewhere in the stream); during render the cropped sub-region
  of the Graphics Object is selected before placement at the composition
  `(x, y)`, matching the whitepaper's Cropping → Palette → Display Image
  pipeline figure. Out-of-range crop coordinates are intersected with
  the source object's real bounds (so a crop that slightly overhangs the
  right or bottom edge becomes the largest sub-rect that exists, and a
  crop landing entirely outside the object paints nothing). Zero-extent
  crops (`w == 0` or `h == 0`) and truncated 8-byte tails are rejected
  at parse time rather than silently producing a blank frame.
- Six new unit tests for the cropping path: a structured-parse assertion
  against a hand-built PCS that an `object_cropped_flag` populates a
  `CropRect { x, y, w, h }` matching the on-wire bytes; a zero-extent
  rejection sweep across `(0,h)`, `(w,0)`, and `(0,0)`; a
  short-by-one-byte truncation rejection test; an end-to-end render
  test asserting that cropping a 4×4 opaque object to `(1, 1, 2, 2)`
  paints only a 2×2 sub-rectangle at the canvas top-left and leaves
  the rest fully transparent; a per-quadrant axis-distinction test that
  exercises both an X-axis crop (top-right quadrant) and a Y-axis crop
  (bottom row) against a 4×3 four-quadrant object and verifies the
  decoded RGBA matches the cropped quadrant; an over-by-the-edge
  clipping test that asserts a crop wider+taller than the object reduces
  to the intersection with object bounds; and an entirely-outside test
  that asserts a crop starting past the object's far edge paints
  nothing (and does not panic).
- VobSub decoder now **applies** the mid-display `CHG_COLCON` palette /
  contrast change command (opcode `0x07`) to the rendered canvas
  rather than length-skipping it. The command's parameter payload is
  parsed into a list of `ChgColConBand` (vertically-bounded `LN_CTLI`)
  + `ChgColConEntry` (horizontal-start-column `PX_CTLI`) values
  exposed on `Spu::chg_colcon`; during canvas paint each pixel inside
  a band's `csln..=ctln` display-line range picks up the right-most
  matching `PX_CTLI`'s replacement palette (4 nibble indices into the
  16-entry `.idx` palette, one per RLE 2-bit pixel value) and
  replacement alpha (4 nibbles in the same order) in lieu of the
  SPU's base `SET_COLOR` / `SET_CONTR` selections. The replacement
  runs rightwards from the entry's `start_col` to the next entry's
  `start_col` or the right edge of the display area. Coordinates are
  in absolute display-line / display-column space (matching the
  bbox-relative `LN_CTLI` / `PX_CTLI` semantics in the SPU spec); the
  decoder maps bitmap-local `(x, y)` back to display space via the
  SPU's `(x1, y1)` origin so bands authored against the full canvas
  apply correctly to SPUs whose bbox doesn't start at `(0, 0)`. The
  parser tolerates payloads with or without the explicit
  `0F FF FF FF` `LN_CTLI` sentinel and rejects truncated `LN_CTLI` /
  `PX_CTLI`, a non-zero reserved high nibble in the `csln` byte,
  `ctln < csln`, and non-strictly-increasing `start_col` values
  inside a band. The `Spu::saw_chg_colcon` flag continues to surface
  the command's presence.
- Eleven new unit tests for the CHG_COLCON application path: a
  bbox-rewriting helper sanity check; structured-parse assertions
  against a hand-built single-band-two-entries payload; a canvas
  mutation test asserting the upper half of an opaque pattern goes
  transparent when a band carries `alpha[pat] = 0` for the matching
  lines; a horizontal-start-column test asserting `PX_CTLI` start
  column boundaries land bitmap-local rather than off-by-one; a
  display-coords-vs-bitmap-local test asserting an SPU rendered at
  `(50, 100)` interprets `csln = 100` and `start_col = 52` as absolute
  display coordinates; truncated-LN_CTLI, truncated-PX_CTLI,
  reserved-high-nibble, inverted-lines, and non-increasing-start-col
  payload negative-input tests; an unterminated-payload
  acceptance test; an overlapping-bands lookup test asserting the
  later band wins; a left-of-start lookup test; and a 200-iteration
  pseudo-random no-panic sweep through `send_packet` →
  `receive_frame`.
- DVB subtitle pixel-line decoders are now exercised by a property +
  negative-input sweep. Each of `decode_2bit_string` /
  `decode_4bit_string` / `decode_8bit_string` gains a round-trip test
  against a hand-built minimal-branch encoder (literal-pixel + short
  zero-pixel + end-of-string forms), a truncated-input test asserting
  `InvalidData` rather than panic, and a 400-iteration pseudo-random
  byte sweep asserting termination + bounded output.
  `parse_pixel_lines` adds tests for end-of-object-line row
  collection, map-table skip (`0x20`/`0x21`/`0x22` introducer +
  two-byte body), truncated-map-table rejection, unknown-data-type
  rejection, and a 400-iteration random-garbage no-panic sweep.
  `read_segment` adds bad-sync-byte rejection, short-header
  `NeedMore`, and truncated-body `NeedMore` tests. End-to-end PES
  round-trips now also cover the 2-bit and 4-bit line blocks
  (previously only the 8-bit branch had end-to-end coverage). Test
  count in `src/dvbsub.rs::tests` rises from 2 to 24; the
  integration test set is unchanged.

### Changed

- Restated the `decode_2bit_string` 01-prefix comment in spec-style
  terms ("01 prefix + 3-bit length + 2-bit colour → (3 + length)
  pixels of the carried colour") in place of an attributive cross-
  reference to other 2-bit-decoder implementations. Behaviour is
  unchanged.

- VobSub `Spu::saw_chg_colcon` flag — present since the previous
  release — is now joined by `Spu::chg_colcon`, the parsed
  `Vec<ChgColConBand>` carrying the mid-display palette / contrast
  replacements (each band has `csln`, `ctln`, and a `Vec` of
  `ChgColConEntry { start_col, palette_sel, alpha }`). The earlier
  length-skip behaviour (added one release ago to stop CHG_COLCON
  from desync-failing the control sequence) is superseded by the
  application path described in the new Added entry above; SPUs that
  carry the command now both decode their base palette / alpha /
  coords / RLE successfully *and* see the requested rectangular
  palette / alpha replacements reach the rendered canvas.

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
