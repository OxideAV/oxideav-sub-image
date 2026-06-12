# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- DVB subtitles: full WRITE direction (ETSI EN 300 743). New public
  segment writers in `dvbsub` — `write_segment` framing,
  `write_display_definition`, `write_page_composition`,
  `write_region_composition` (via `RegionCompositionDef`: fill flag,
  per-depth fill pixel codes, pixel-coded object placements),
  `write_clut_definition` (via `ClutEntryDef`: 4-byte full-range and
  packed 2-byte forms) and `write_object_data` (top/bottom field
  split, optional map-table prefixes) — each the inverse of the
  corresponding decoder parse. True run-length 2/4/8-bit
  pixel-code-string encoders (`encode_2bit_pixel_string` /
  `encode_4bit_pixel_string` / `encode_8bit_pixel_string`) cover every
  counted-run form, the single-pixel-of-colour-0 codes, end-of-string
  termination and byte alignment, with runs longer than a form's
  maximum chunked greedily. `rgba_to_clut_ycbcrt` is the integer
  inverse of the decode-side BT.601 CLUT transform (greys bit-exact,
  colours within ±2 LSBs, alpha exact). `dvbsub::make_encoder` (also
  registered through `register_codecs`) turns RGBA frames into
  complete display-set PES payloads — DDS + PCS + RCS + CLUT + ODS +
  END — picking the smallest pixel depth that fits the quantised
  palette (≤ 4 entries → 2-bit, ≤ 16 → 4-bit, else 8-bit, with a
  3-3-2-2 reduction fallback past 255 distinct colours), cropping to
  the tight bounding box of non-transparent pixels, and emitting an
  erase page (no regions) for a fully-transparent frame. All writers
  are pinned by encode→decode roundtrips through the existing decode
  path: per-form run-length boundary sweeps, randomised rows,
  even/odd/single-row field interleave, map-table transparency, CLUT
  short-vs-full form agreement, and nine end-to-end encoder→decoder
  integration tests.

- DVB subtitle multi-region compositing is now covered by a regression
  test (`dvbsub::tests::multi_region_overlap_composites_in_page_order`):
  a display set that references two overlapping regions in the
  page-composition segment paints them in list order with Porter–Duff
  source-over, so a later region carrying a partially-transparent CLUT
  entry blends over the region beneath it rather than discarding it.

### Changed

- Corrected the DVB subtitle module/README documentation: multi-region
  displays composite in page-composition list order (each region over
  the previous, source-over), not "first region wins" as the previous
  note implied. The decoder already behaved this way; only the prose
  was stale.

### Added (PGS)

- PGS Palette Definition Segments are now kept in independent slots
  keyed by their on-wire `palette_id` byte, instead of folding all
  PDS entries into a single shared 256-entry table. The render path
  consults the PCS's `palette_id` field to pick which slot to render
  against: a display-set that carries one PDS at `palette_id == 0`
  and a second PDS at `palette_id == 1` (the shape an authoring tool
  uses for a colour-change or fade effect, where the next epoch's
  PCS will reference the other slot to swap palettes without
  reloading the bitmap) now leaves the two palettes side-by-side
  rather than letting the second PDS overwrite the first. Before
  this change a PCS referencing slot 0 would see slot 1's last-PDS
  bytes — silently wrong for any stream that authored more than one
  palette per set. The keyed map preserves the existing entry-delta
  rule for a same-id repeat (a second PDS for the same `palette_id`
  adds / replaces individual entries on top of whatever the first
  one wrote). A PCS whose `palette_id` references a slot no PDS
  populated falls back to the default (all-transparent) palette, so
  the render stays defined on malformed streams without panicking.
- Five new unit tests in `src/pgs.rs::tests` cover the keying path:
  a structured-parse assertion that `parse_pds_into` surfaces
  `palette_id` and routes two ids into independent slots; a
  short-body rejection test; a same-id additive-merge test
  confirming a second PDS for an existing slot adds entry deltas
  without clobbering earlier ones; an end-to-end render test
  asserting the same dual-PDS display-set rendered with PCS
  `palette_id == 0` vs. `palette_id == 1` produces visibly
  different pixels (red-dominant vs. blue-dominant); and a
  missing-id render test asserting a PCS pointing at an unpopulated
  slot renders fully transparent rather than crashing.
- `pgs::WindowDefinition` and the `pgs::parse_wds` helper expose the
  Window Definition Segment as a typed `Vec<WindowDefinition>` with
  `(window_id, x, y, w, h)` in canvas-pixel space. The parser rejects
  empty bodies, a declared window count that does not match the
  remaining body bytes exactly, and zero-extent windows.
- The PGS decoder now retains the WDS window table keyed by
  `window_id` and clips each composition object's paint rectangle to
  the intersection of the canvas and its assigned window. Pixels that
  would land outside the window are dropped before the blit; objects
  that land entirely outside paint nothing. A display-set whose WDS
  carries zero windows (the canonical "erase" form) preserves the
  prior whole-canvas paint area for backward compatibility.
- VobSub SPU control-sequence traversal now latches `Spu::start_delay_raw`
  from the `SP_DCSQ_STM` of the DCSQ that actually carries the
  `STA_DSP` (`0x01`) or `FSTA_DSP` (`0x00`) command, instead of
  unconditionally locking onto the first DCSQ's STM. Typical SPUs put
  palette / alpha / `SET_DAREA` / `SET_DSPXA` in DCSQ #0 with
  `STM = 0` and schedule the on-display event in DCSQ #1 with a
  non-zero STM (the delay before the cue appears); the earlier
  first-DCSQ rule reported a permanently zero start delay on that
  common shape. The first start-display command encountered wins;
  redundant retriggers in later DCSQs are tolerated but do not
  overwrite the latched delay.
- `Spu::forced_display: bool` surfaces the SPU control sequence's
  `FSTA_DSP` (Forced Start Display, command `0x00`). A forced
  subtitle is one a player should display even when subtitles are
  otherwise disabled — typically used for translations of on-screen
  signs / foreign-language dialogue inside an otherwise untranslated
  soundtrack. The flag captures presence anywhere in the DCSQ chain
  and is independent of `STA_DSP` / `STP_DSP`. The previous
  behaviour silently dropped FSTA_DSP on the floor; consumers had no
  way to tell a forced cue from a regular one.
- `STP_DSP` (`0x02`) now overwrites `Spu::stop_delay_raw` on every
  occurrence rather than capturing only the first; when an authoring
  tool revises the end time inside a later DCSQ, the revised value
  wins. The change preserves the *latest stop wins* semantics
  documented for the command and lines the field up with the new
  STA_DSP first-encountered-wins rule on the start side.
- The DCSQ-chain walker now bails when `SP_NXT_DCSQ_SA` does not
  advance the cursor — the spec's "if this is the last SP_DCSQ, it
  points to itself" rule — guarding against an infinite loop on
  malformed back-pointers. Behaviour-preserving for well-formed SPUs.
- Eight new unit tests in `src/vobsub.rs::tests`: a sanity check that
  the new `build_spu_with_dcsq_chain` helper produces a decodable
  setup-only SPU with `start_delay_raw == 0` and
  `forced_display == false`; a regression test that an STA_DSP in
  the second DCSQ latches `start_delay_raw` to that DCSQ's STM and
  not to the first DCSQ's STM (the headline shape this commit
  fixes); a forced-display test confirming FSTA_DSP raises
  `forced_display` and latches `start_delay_raw` to its DCSQ's STM;
  an STP_DSP last-write-wins assertion across two stop-bearing
  DCSQs; a first-start-wins test across two STA_DSP-bearing DCSQs;
  a mixed-order test asserting that STA_DSP followed by FSTA_DSP
  still raises `forced_display` without re-latching the start delay;
  a 16-deep self-pointer-terminator chain that confirms the
  traversal does not spin; and an unknown-command rejection test
  inside a non-first DCSQ to lock in the post-`first_seq` error
  shape.

- PGS PCS `composition_state` / `palette_update_flag` / `palette_id`
  bytes are now parsed onto `PresentationComposition` instead of
  silently dropped. Three named constants
  (`COMP_STATE_NORMAL = 0x00`, `COMP_STATE_ACQUISITION = 0x40`,
  `COMP_STATE_EPOCH_START = 0x80`) and an `is_random_access(state)`
  helper let consumers classify a display-set against random-access
  semantics without inspecting raw bytes. The PGS `.sup` demuxer
  routes that classification onto `Packet::flags.keyframe`: only sets
  whose PCS reports `Acquisition Point` or `Epoch Start` are flagged
  as keyframes, so a seeker that rounds to the nearest keyframe lands
  on a display-set that decodes standalone. Mid-epoch `Normal Case`
  packets and packets carrying an unrecognised composition_state byte
  stay un-flagged. Encoder behaviour is unchanged — every emitted
  display-set is still a self-contained `Epoch Start`, so its packets
  round-trip back to `keyframe = true` on demux.
- Nine new unit tests for the PCS-classification path: structured-parse
  assertions that `composition_state` / `palette_update_flag` / `palette_id`
  surface verbatim through `parse_pcs`; assertions that `Acquisition
  Point` and `Epoch Start` report as random-access while `Normal Case`
  and an out-of-range byte do not; a `palette_update_flag` lower-bits
  test confirming reserved bits do not flip the flag; demuxer
  classification tests asserting that a five-set stream interleaving
  all three composition_states yields `[true, false, true, false,
  false]` keyframe flags, that an `Acquisition Point` followed by a
  `Normal Case` set drops the flag back to false (state must not be
  sticky across display-set boundaries), and that an unknown
  composition_state byte does not promote a packet to a keyframe; plus
  an encoder-output round-trip asserting `make_encoder` emits
  `COMP_STATE_EPOCH_START` on every set.

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

- DVB subtitle `region_composition_segment` now honours
  `region_fill_flag` (ETSI EN 300 743 §7.2.3). When the flag is set,
  the region rectangle is pre-painted with the depth-appropriate
  `region_n-bit_pixel_code` (8/4/2-bit), translated through the
  region's CLUT, *before* any objects composite on top — matching the
  spec's "fill the region area with the colour given by the n-bit
  pixel code" wording. Cleared, the rectangle stays at the canvas's
  transparent background, exactly as before. The pre-fill rectangle
  is clipped to the canvas, so a region declared past the right or
  bottom edge writes only its in-bounds intersection (no buffer
  overrun on malformed streams). The byte layout the parser already
  inspected to advance past the three pixel-code bytes is now
  preserved on the typed `Region` (`fill: bool`,
  `fill_code_8 / fill_code_4 / fill_code_2: u8`), removing the
  `#[allow(dead_code)]` on `width` / `height` / `depth_bits`. Five
  new unit tests cover the path: a structured-parse assertion that
  `region_fill_flag = 1` + the three pixel-code bytes decode to the
  expected typed fields; the cleared-flag counterpart asserting `fill
  = false` survives a non-zero pixel-code byte; an end-to-end render
  test against a 4×3 canvas with a 2×2 fill region at (1, 0) +
  CLUT-1 = white, asserting the four region pixels go opaque white
  and the six off-region pixels stay transparent; the
  cleared-flag end-to-end test asserting an identical RCS but with
  the flag cleared leaves the whole canvas transparent (the
  renderer keys off `fill`, not the CLUT entry); and a clip-to-canvas
  test placing a 4×4 region at (3, 1) against a 4×3 canvas and
  asserting only the in-bounds 1×2 strip is painted.

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
