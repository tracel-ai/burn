use super::*;

/// Build a `[rows, cols]` row-major boolean mask whose rows exercise every
/// logical-reduction merge case along the last axis: all-zero, all-nonzero, a
/// single nonzero at the first/last/middle column, and an all-nonzero row with
/// a single zero. Used by the large `any_dim`/`all_dim` tests (`rows == 8`).
pub fn build_logical_mask(rows: usize, cols: usize) -> Vec<bool> {
    assert!(
        rows >= 8 && cols > 100,
        "mask layout assumes rows>=8, cols>100"
    );
    let mut mask = vec![false; rows * cols];
    for c in 0..cols {
        mask[cols + c] = true; // row 1: all nonzero
        mask[5 * cols + c] = true; // row 5: all nonzero
        mask[7 * cols + c] = true; // row 7: all nonzero (one zero punched below)
    }
    mask[2 * cols] = true; // row 2: single nonzero at the first column
    mask[3 * cols + (cols - 1)] = true; // row 3: single nonzero at the last column
    mask[6 * cols + cols / 2] = true; // row 6: single nonzero in the middle
    mask[7 * cols + 100] = false; // row 7: a single zero deep inside an all-nonzero row
    // rows 0 and 4 stay all-zero
    mask
}

/// Map the mask to float input, using alternating nonzero magnitudes so the
/// reduction must normalize arbitrary nonzero values (not just `1.0`) to true.
pub fn mask_to_floats(mask: &[bool]) -> Vec<f32> {
    mask.iter()
        .enumerate()
        .map(|(i, &m)| {
            if m {
                if i % 2 == 0 { 2.0 } else { -3.0 }
            } else {
                0.0
            }
        })
        .collect()
}

/// Map the mask to int input with alternating nonzero magnitudes.
pub fn mask_to_ints(mask: &[bool]) -> Vec<i32> {
    mask.iter()
        .enumerate()
        .map(|(i, &m)| {
            if m {
                if i % 2 == 0 { 2 } else { -3 }
            } else {
                0
            }
        })
        .collect()
}

mod abs;
mod add;
mod aggregation;
mod all;
mod any;
mod arg;
mod blackman_window;
mod cast;
mod cat;
mod categorical;
mod ceil;
mod chunk;
mod clamp;
mod close;
mod comparison;
mod create_like;
mod cross;
mod cumulative;
mod div;
mod dot;
mod erf;
mod exp;
mod expand;
mod finite;
mod flatten;
mod flip;
mod floor;
mod fmod;
mod full;
mod gather_scatter;
mod gather_scatter_nd;
mod grid_sample;
mod hamming_window;
mod hann_window;
mod hypot;
mod inf;
mod init;
mod iter_dim;
mod log;
mod log1p;
mod mask;
mod matmul;
mod maxmin;
mod movedim;
mod mul;
mod nan;
mod narrow;
mod neg;
mod one_hot;
mod padding;
mod permute;
mod powf;
mod powf_scalar;
mod prod;
mod random;
mod recip;
mod remainder;
mod repeat;
mod repeat_dim;
mod reshape;
mod round;
mod select;
mod sign;
mod slice;
mod slice_assign;
mod sort_argsort;
mod split;
mod sqrt;
mod square;
mod squeeze;
mod stack;
mod sub;
mod take;
mod topk;
mod transaction;
mod transpose;
mod tri;
mod trig;
mod trunc;
mod unfold;
