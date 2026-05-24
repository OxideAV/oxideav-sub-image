# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
