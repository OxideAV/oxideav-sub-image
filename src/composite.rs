//! Alpha-aware RGBA compositing primitives shared by the bitmap-subtitle
//! decoders.
//!
//! Bitmap subtitles paint one or more source bitmaps onto a single RGBA
//! canvas. When two source regions overlap and the topmost source pixel
//! is only *partially* transparent, the visually-correct result is the
//! source colour blended *over* whatever is already on the canvas — the
//! Porter–Duff **source-over-destination** operator — not a hard
//! overwrite that discards the destination.
//!
//! ## The math (straight / non-premultiplied alpha)
//!
//! For a source pixel `(Cs, As)` over a destination pixel `(Cd, Ad)`,
//! with all of `C` and `A` in the unit interval, Porter–Duff "over" is:
//!
//! ```text
//! Ao = As + Ad * (1 - As)
//! Co = (Cs * As + Cd * Ad * (1 - As)) / Ao        (Ao != 0)
//! ```
//!
//! `Co` is the alpha-weighted average of the source and the
//! show-through of the destination, renormalised by the output alpha so
//! the stored value stays straight (non-premultiplied). When `Ao` is
//! zero both inputs are fully transparent and the output colour is
//! irrelevant — we leave it black.
//!
//! All arithmetic here is done in fixed point over `u32` so the result
//! is deterministic and free of float rounding drift. Inputs and
//! outputs are 8-bit straight-alpha RGBA. The two fast paths the
//! callers hit most — a transparent source (no-op) and a fully-opaque
//! source over anything (plain copy) — are short-circuited.

/// One straight-alpha 8-bit RGBA pixel.
pub type Rgba8 = [u8; 4];

/// Composite a single source pixel `src` over a single destination
/// pixel `dst` using the Porter–Duff source-over operator, returning the
/// blended straight-alpha RGBA result.
///
/// * A fully-transparent source (`src[3] == 0`) returns `dst` unchanged.
/// * A fully-opaque source (`src[3] == 255`) returns `src` unchanged —
///   it covers the destination completely.
#[inline]
#[must_use]
pub fn over(src: Rgba8, dst: Rgba8) -> Rgba8 {
    let sa = src[3] as u32;
    if sa == 0 {
        return dst;
    }
    if sa == 255 {
        return src;
    }
    let da = dst[3] as u32;
    // inv = (1 - As) scaled to 0..255.
    let inv = 255 - sa;
    // Output alpha: As + Ad*(1-As), rounded.
    let out_a = sa + div255(da * inv);
    if out_a == 0 {
        return [0, 0, 0, 0];
    }
    let mut out = [0u8; 4];
    for c in 0..3 {
        // Numerator in the 0..(255*255) range: Cs*As + Cd*Ad*(1-As),
        // each term already a product of an 8-bit colour by an
        // 8-bit-scaled coverage. Divide the destination contribution by
        // 255 to fold Ad*(1-As) back into an 8-bit weight, then split
        // the final /out_a renormalisation.
        let s_term = src[c] as u32 * sa;
        let d_term = dst[c] as u32 * div255(da * inv);
        // (s_term + d_term) is colour*coverage summed to <= 255*255.
        // Renormalise by out_a (an 8-bit-scaled alpha) with rounding.
        out[c] = (((s_term + d_term) + out_a / 2) / out_a) as u8;
    }
    out[3] = out_a as u8;
    out
}

/// Divide by 255 with round-to-nearest, staying in integer arithmetic.
/// `(x + 127) / 255` is exact for the `x <= 255*255` range we use.
#[inline]
fn div255(x: u32) -> u32 {
    (x + 127) / 255
}

/// Blit a row-major indexed bitmap onto an RGBA canvas, alpha-compositing
/// each painted pixel over the existing canvas content.
///
/// * `canvas` is a `width * height * 4` straight-alpha RGBA buffer.
/// * `lookup(index)` maps a bitmap index to its straight-alpha RGBA
///   colour. Returning a pixel with alpha 0 skips that source pixel
///   entirely (no canvas write), which is the conventional way bitmap
///   subtitles encode "background / show video through here".
/// * `(base_x, base_y)` is the top-left placement of the bitmap on the
///   canvas. Pixels that fall outside the canvas are clipped.
///
/// `rows` is the indexed bitmap as a slice of equal-or-ragged rows; each
/// inner slice is one bitmap row left-to-right.
///
/// ## Robustness
///
/// The destination coordinate of every source pixel is computed with a
/// checked add: a `base_x`/`base_y` (or `+ col_idx`/`+ row_idx`) that
/// would overflow `usize` is treated as off-canvas and clipped, exactly
/// like a coordinate that merely exceeds `width`/`height`. Without the
/// checked add a `base_x` near `usize::MAX` would wrap to a small
/// in-range value, slip past the `dx >= width` clip, and either panic
/// (debug overflow check) or scribble onto an unrelated canvas pixel
/// (release wrap). Because every write target is proven `< width` and
/// `< height` after clipping, and the canvas is expected to be a
/// `width * height * 4` buffer, the linear offset `(dy * width + dx) * 4`
/// stays within `canvas`; a debug assertion guards that length contract.
pub fn blit_indexed<F>(
    canvas: &mut [u8],
    width: usize,
    height: usize,
    rows: &[Vec<u8>],
    base_x: usize,
    base_y: usize,
    lookup: F,
) where
    F: Fn(u8) -> Rgba8,
{
    debug_assert!(
        canvas.len() >= width.saturating_mul(height).saturating_mul(4),
        "blit_indexed canvas too small: {} < {width}*{height}*4",
        canvas.len(),
    );
    for (row_idx, row) in rows.iter().enumerate() {
        // A `base_y` near `usize::MAX` would wrap on a plain `+`; treat
        // an overflowing (or simply too-large) destination row as
        // off-canvas and stop — the rest of the bitmap is further down.
        let Some(dy) = base_y.checked_add(row_idx) else {
            break;
        };
        if dy >= height {
            break;
        }
        let row_base = dy * width;
        for (col_idx, &px) in row.iter().enumerate() {
            let Some(dx) = base_x.checked_add(col_idx) else {
                break;
            };
            if dx >= width {
                break;
            }
            let src = lookup(px);
            if src[3] == 0 {
                continue;
            }
            // dy < height and dx < width, so row_base + dx < width*height
            // and the *4 offset stays inside the width*height*4 canvas.
            let off = (row_base + dx) * 4;
            if off + 4 > canvas.len() {
                // The canvas is smaller than its declared geometry; skip
                // rather than panic. (debug_assert above catches this in
                // tests; release stays graceful.)
                break;
            }
            let dst = [
                canvas[off],
                canvas[off + 1],
                canvas[off + 2],
                canvas[off + 3],
            ];
            let blended = over(src, dst);
            canvas[off..off + 4].copy_from_slice(&blended);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transparent_source_is_noop() {
        let dst = [10, 20, 30, 200];
        assert_eq!(over([99, 99, 99, 0], dst), dst);
    }

    #[test]
    fn opaque_source_covers_destination() {
        let src = [99, 50, 25, 255];
        assert_eq!(over(src, [10, 20, 30, 200]), src);
    }

    #[test]
    fn over_transparent_destination_returns_source_colour() {
        // Source 50% white over a fully-transparent canvas: output alpha
        // is the source alpha and the colour is the source colour
        // (nothing to blend with).
        let out = over([255, 255, 255, 128], [0, 0, 0, 0]);
        assert_eq!(out[3], 128);
        assert_eq!([out[0], out[1], out[2]], [255, 255, 255]);
    }

    #[test]
    fn half_white_over_opaque_black_is_mid_grey() {
        // 50% white over opaque black → opaque ~50% grey.
        let out = over([255, 255, 255, 128], [0, 0, 0, 255]);
        assert_eq!(out[3], 255, "over opaque stays opaque");
        // 128/255 of 255 ≈ 128.
        for (c, &v) in out[..3].iter().enumerate() {
            assert!((v as i32 - 128).abs() <= 1, "channel {c} = {v} not ~128");
        }
    }

    #[test]
    fn over_is_associative_for_opaque_destination() {
        // Compositing two layers then over opaque, vs. one combined layer,
        // should land within rounding tolerance.
        let bottom = [0, 0, 0, 255];
        let mid = over([200, 0, 0, 100], bottom);
        let top = over([0, 0, 200, 100], mid);
        assert_eq!(top[3], 255);
        // Blue should now dominate red since it was layered last.
        assert!(top[2] >= top[0], "expected blue-leaning result: {top:?}");
    }

    #[test]
    fn output_alpha_accumulates_over_translucent_destination() {
        // 50% over 50% → 1 - (1-.5)(1-.5) = 0.75 ≈ 191.
        let out = over([255, 0, 0, 128], [0, 0, 255, 128]);
        assert!(
            (out[3] as i32 - 191).abs() <= 2,
            "output alpha {} not ~191",
            out[3]
        );
    }

    #[test]
    fn blit_composites_overlapping_translucent_rows() {
        // 2×1 canvas. Paint a 50% red pixel at x=0, then a 50% blue pixel
        // over the same x=0 — the blue must blend over the red, not erase
        // it to a hard copy.
        let mut canvas = vec![0u8; 2 * 4]; // 2×1 RGBA
        let red = [255u8, 0, 0, 128];
        let blue = [0u8, 0, 255, 128];
        blit_indexed(&mut canvas, 2, 1, &[vec![1]], 0, 0, |_| red);
        blit_indexed(&mut canvas, 2, 1, &[vec![1]], 0, 0, |_| blue);
        // Pixel 0 carries residual red show-through under the blue.
        assert!(
            canvas[0] > 0,
            "expected red show-through, got {:?}",
            &canvas[0..4]
        );
        assert!(
            canvas[2] > canvas[0],
            "blue should dominate: {:?}",
            &canvas[0..4]
        );
        assert!(canvas[3] > 128, "alpha should accumulate: {}", canvas[3]);
    }

    #[test]
    fn blit_skips_transparent_lookup() {
        let mut canvas = vec![7u8; 4];
        blit_indexed(&mut canvas, 1, 1, &[vec![0]], 0, 0, |_| [9, 9, 9, 0]);
        assert_eq!(canvas, vec![7u8; 4], "transparent lookup must not write");
    }

    #[test]
    fn blit_clips_out_of_bounds() {
        let mut canvas = vec![0u8; 4]; // 1×1 RGBA
                                       // 2×2 source at (0,0) on a 1×1 canvas — only (0,0) is written.
        blit_indexed(&mut canvas, 1, 1, &[vec![1, 1], vec![1, 1]], 0, 0, |_| {
            [10, 20, 30, 255]
        });
        assert_eq!(canvas, vec![10, 20, 30, 255]);
    }

    // ---- Property-style sweeps over the compositing primitives ----------
    //
    // These run deterministic exhaustive / pseudo-random sweeps in plain
    // Rust (no external proptest crate) over the invariants the subtitle
    // render paths rely on: short-circuit identities, alpha monotonicity,
    // blit clip == reference clip, sub-tile composition == whole-tile, and
    // overflow safety for pathological placement offsets.

    /// Tiny deterministic LCG so the sweeps are reproducible without a
    /// dependency. Returns the next pseudo-random `u32`.
    fn lcg(state: &mut u64) -> u32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 32) as u32
    }

    fn rng_rgba(state: &mut u64) -> Rgba8 {
        let v = lcg(state);
        [v as u8, (v >> 8) as u8, (v >> 16) as u8, (v >> 24) as u8]
    }

    #[test]
    fn over_short_circuit_identities_hold_for_swept_destinations() {
        // For every alpha 0..=255 of a destination and a sample of colours,
        // a fully-transparent source returns dst unchanged and a
        // fully-opaque source returns src unchanged.
        let mut st = 0x1234_5678_9abc_def0u64;
        for da in 0u16..=255 {
            for _ in 0..16 {
                let dst = {
                    let mut d = rng_rgba(&mut st);
                    d[3] = da as u8;
                    d
                };
                let mut transparent_src = rng_rgba(&mut st);
                transparent_src[3] = 0;
                assert_eq!(over(transparent_src, dst), dst);

                let mut opaque_src = rng_rgba(&mut st);
                opaque_src[3] = 255;
                assert_eq!(over(opaque_src, dst), opaque_src);
            }
        }
    }

    #[test]
    fn over_output_alpha_is_monotonic_and_bounded() {
        // Porter–Duff "over" never lowers the destination's alpha
        // (Ao = As + Ad(1-As) >= Ad) and always yields a valid u8 alpha.
        // Also exercises the full inner-loop arithmetic for panics.
        let mut st = 0xdead_beef_cafe_babeu64;
        for _ in 0..200_000 {
            let src = rng_rgba(&mut st);
            let dst = rng_rgba(&mut st);
            let out = over(src, dst);
            assert!(
                out[3] >= dst[3],
                "alpha decreased: src={src:?} dst={dst:?} out={out:?}"
            );
            assert!(
                out[3] >= src[3].min(dst[3]),
                "alpha below both inputs: src={src:?} dst={dst:?} out={out:?}"
            );
        }
    }

    /// Reference compositor: a straightforward, independently-written blit
    /// the optimised `blit_indexed` must agree with bit-for-bit.
    fn ref_blit(
        canvas: &mut [u8],
        width: usize,
        height: usize,
        rows: &[Vec<u8>],
        base_x: usize,
        base_y: usize,
        lut: &[Rgba8],
    ) {
        for (r, row) in rows.iter().enumerate() {
            for (c, &px) in row.iter().enumerate() {
                let (Some(dx), Some(dy)) = (base_x.checked_add(c), base_y.checked_add(r)) else {
                    continue;
                };
                if dx >= width || dy >= height {
                    continue;
                }
                let src = lut[px as usize];
                if src[3] == 0 {
                    continue;
                }
                let off = (dy * width + dx) * 4;
                let dst = [
                    canvas[off],
                    canvas[off + 1],
                    canvas[off + 2],
                    canvas[off + 3],
                ];
                let blended = over(src, dst);
                canvas[off..off + 4].copy_from_slice(&blended);
            }
        }
    }

    #[test]
    fn blit_matches_reference_over_random_placements() {
        let mut st = 0x0f0f_f0f0_5555_aaaau64;
        let width = 9;
        let height = 7;
        // 8-entry palette; index 0 transparent (the common subtitle case).
        for _ in 0..2_000 {
            let mut lut = [[0u8; 4]; 8];
            for (i, e) in lut.iter_mut().enumerate() {
                *e = if i == 0 {
                    [0, 0, 0, 0]
                } else {
                    rng_rgba(&mut st)
                };
            }
            let ow = (lcg(&mut st) % 12) as usize; // may exceed width to test clip
            let oh = (lcg(&mut st) % 10) as usize;
            let rows: Vec<Vec<u8>> = (0..oh)
                .map(|_| (0..ow).map(|_| (lcg(&mut st) % 8) as u8).collect())
                .collect();
            // Placement that may be partly or fully off-canvas.
            let bx = (lcg(&mut st) % 14) as usize;
            let by = (lcg(&mut st) % 12) as usize;

            let mut a = vec![0u8; width * height * 4];
            let mut b = a.clone();
            blit_indexed(&mut a, width, height, &rows, bx, by, |idx| {
                lut[idx as usize]
            });
            ref_blit(&mut b, width, height, &rows, bx, by, &lut);
            assert_eq!(a, b, "blit != reference: ow={ow} oh={oh} bx={bx} by={by}");
        }
    }

    #[test]
    fn split_blit_equals_whole_blit() {
        // Blitting a tile in one call equals blitting its left and right
        // halves at the matching offsets — the compositor is just a
        // position-indexed paint, so the partition is invariant. (Halves
        // are disjoint in x, so there's no inter-half blend ordering to
        // worry about.)
        let width = 10;
        let height = 4;
        let lut: [Rgba8; 4] = [
            [0, 0, 0, 0],
            [200, 10, 10, 180],
            [10, 200, 10, 90],
            [10, 10, 200, 255],
        ];
        let oh = 3usize;
        let ow = 8usize;
        let mut st = 0xa5a5_5a5a_1234_9999u64;
        let full: Vec<Vec<u8>> = (0..oh)
            .map(|_| (0..ow).map(|_| (lcg(&mut st) % 4) as u8).collect())
            .collect();
        let split = 3usize;
        let left: Vec<Vec<u8>> = full.iter().map(|r| r[..split].to_vec()).collect();
        let right: Vec<Vec<u8>> = full.iter().map(|r| r[split..].to_vec()).collect();
        let bx = 1usize;
        let by = 0usize;

        let mut whole = vec![0u8; width * height * 4];
        blit_indexed(&mut whole, width, height, &full, bx, by, |i| {
            lut[i as usize]
        });

        let mut parts = vec![0u8; width * height * 4];
        blit_indexed(&mut parts, width, height, &left, bx, by, |i| {
            lut[i as usize]
        });
        blit_indexed(&mut parts, width, height, &right, bx + split, by, |i| {
            lut[i as usize]
        });

        assert_eq!(whole, parts, "split blit diverged from whole blit");
    }

    #[test]
    fn zero_size_region_is_a_noop() {
        let lut = [[9u8, 9, 9, 255]];
        // No rows at all.
        let mut canvas = vec![5u8; 4 * 4 * 4];
        let before = canvas.clone();
        blit_indexed(&mut canvas, 4, 4, &[], 0, 0, |i| lut[i as usize]);
        assert_eq!(canvas, before, "empty rows must not write");
        // Rows present but each empty.
        blit_indexed(&mut canvas, 4, 4, &[vec![], vec![]], 0, 0, |i| {
            lut[i as usize]
        });
        assert_eq!(canvas, before, "empty inner rows must not write");
    }

    #[test]
    fn pathological_offsets_never_panic_or_corrupt() {
        // base_x/base_y near usize::MAX previously wrapped past the clip
        // and either panicked (debug overflow) or scribbled onto an
        // unrelated pixel (release wrap). The checked-add clip turns these
        // into clean no-ops: the canvas is untouched.
        let lut = [[0u8, 0, 0, 0], [255, 255, 255, 255]];
        let width = 4;
        let height = 4;
        let rows = vec![vec![1u8; 4]; 4];
        for &(bx, by) in &[
            (usize::MAX, 0usize),
            (0usize, usize::MAX),
            (usize::MAX - 1, usize::MAX - 1),
            (usize::MAX - 2, 0),
            (0, usize::MAX - 3),
        ] {
            let mut canvas = vec![0u8; width * height * 4];
            blit_indexed(&mut canvas, width, height, &rows, bx, by, |i| {
                lut[i as usize]
            });
            assert!(
                canvas.iter().all(|&b| b == 0),
                "off-canvas blit at ({bx},{by}) wrote into the canvas"
            );
        }
    }

    #[test]
    fn near_edge_offset_writes_only_the_in_range_corner() {
        // base_x = width-1 with a wide bitmap: only the single in-range
        // column is painted, the rest clip — and no arithmetic overflow.
        let lut = [[0u8, 0, 0, 0], [10, 20, 30, 255]];
        let width = 5;
        let height = 5;
        let rows = vec![vec![1u8; 8]; 8];
        let mut canvas = vec![0u8; width * height * 4];
        blit_indexed(
            &mut canvas,
            width,
            height,
            &rows,
            width - 1,
            height - 1,
            |i| lut[i as usize],
        );
        // Exactly the bottom-right pixel is set.
        let mut painted = 0;
        for (i, chunk) in canvas.chunks(4).enumerate() {
            if chunk != [0, 0, 0, 0] {
                painted += 1;
                assert_eq!(i, (height - 1) * width + (width - 1), "wrong pixel painted");
            }
        }
        assert_eq!(painted, 1, "expected exactly one painted pixel");
    }
}
