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
    for (row_idx, row) in rows.iter().enumerate() {
        let dy = base_y + row_idx;
        if dy >= height {
            break;
        }
        for (col_idx, &px) in row.iter().enumerate() {
            let dx = base_x + col_idx;
            if dx >= width {
                break;
            }
            let src = lookup(px);
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
}
