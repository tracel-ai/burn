//! Portable SIMD kernels using macerator.
//!
//! These kernels work across all architectures supported by macerator:
//! - aarch64 (NEON)
//! - x86_64 (AVX2, AVX512, SSE)
//! - wasm32 (SIMD128)
//! - Scalar fallback for embedded/other platforms

use core::iter::Sum;
use core::ops::AddAssign;

use macerator::{
    ReduceAdd, ReduceMax, ReduceMin, Simd, VAdd, VOrd, vload_unaligned, vstore_unaligned,
};

// ============================================================================
// Sum reduction
// ============================================================================

/// Sum all elements in a f32 slice using SIMD with 4 accumulators.
#[inline]
pub fn sum_f32(data: &[f32]) -> f32 {
    macerator_sum(data)
}

/// 8-accumulator SIMD sum. Independent accumulator chains let the CPU
/// pipeline floating-point adds and hide L2 cache latency.
#[macerator::with_simd]
fn macerator_sum<S: Simd, F: VAdd + Sum + ReduceAdd>(mut xs: &[F]) -> F {
    let lanes = F::lanes::<S>();
    let stride = lanes * 8;
    let zero = F::default().splat::<S>();
    let (mut s0, mut s1, mut s2, mut s3) = (zero, zero, zero, zero);
    let (mut s4, mut s5, mut s6, mut s7) = (zero, zero, zero, zero);

    while xs.len() >= stride {
        unsafe {
            let p = xs.as_ptr();
            s0 += vload_unaligned(p);
            s1 += vload_unaligned(p.add(lanes));
            s2 += vload_unaligned(p.add(lanes * 2));
            s3 += vload_unaligned(p.add(lanes * 3));
            s4 += vload_unaligned(p.add(lanes * 4));
            s5 += vload_unaligned(p.add(lanes * 5));
            s6 += vload_unaligned(p.add(lanes * 6));
            s7 += vload_unaligned(p.add(lanes * 7));
        }
        xs = &xs[stride..];
    }

    // Combine 8 accumulators into one, then drain remaining full vectors
    let mut sum = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
    while xs.len() >= lanes {
        sum += unsafe { vload_unaligned(xs.as_ptr()) };
        xs = &xs[lanes..];
    }

    sum.reduce_add() + xs.iter().copied().sum()
}

// ============================================================================
// Scatter-add for dimension reductions
// ============================================================================

/// Scatter-add: for each row, add to corresponding output positions.
/// Used for cache-friendly first-dim and middle-dim reductions.
///
/// # Arguments
/// * `src` - Source data pointer
/// * `dst` - Destination accumulator (must be pre-zeroed)
/// * `num_rows` - Number of rows to sum
/// * `row_len` - Length of each row (columns)
/// * `src_row_stride` - Stride between source rows
#[macerator::with_simd]
pub fn scatter_add_f32<S: Simd, F: VAdd + AddAssign>(
    src: &[F],
    dst: &mut [F],
    num_rows: usize,
    row_len: usize,
    src_row_stride: usize,
) {
    let lanes = F::lanes::<S>();

    for row in 0..num_rows {
        let row_start = row * src_row_stride;
        let row_data = &src[row_start..row_start + row_len];

        let simd_len = row_len / lanes * lanes;

        // SIMD accumulate
        let mut i = 0;
        while i < simd_len {
            unsafe {
                let s = vload_unaligned(row_data.as_ptr().add(i));
                let d = vload_unaligned(dst.as_ptr().add(i));
                vstore_unaligned::<S, _>(dst.as_mut_ptr().add(i), d + s);
            }
            i += lanes;
        }

        // Scalar tail
        for j in simd_len..row_len {
            dst[j] += row_data[j];
        }
    }
}

/// Batched scatter-add for middle-dim reductions.
/// For tensors like [B, M, K] reducing dim=1.
#[macerator::with_simd]
pub fn scatter_add_batched<S: Simd, F: VAdd + AddAssign>(
    src: &[F],
    dst: &mut [F],
    num_batches: usize,
    num_rows: usize,
    row_len: usize,
    batch_stride: usize,
    row_stride: usize,
) {
    let lanes = F::lanes::<S>();

    for batch in 0..num_batches {
        let batch_src_start = batch * batch_stride;
        let batch_dst_start = batch * row_len;
        let batch_dst = &mut dst[batch_dst_start..batch_dst_start + row_len];

        for row in 0..num_rows {
            let row_start = batch_src_start + row * row_stride;
            let row_data = &src[row_start..row_start + row_len];

            let simd_len = row_len / lanes * lanes;

            let mut i = 0;
            while i < simd_len {
                unsafe {
                    let s = vload_unaligned(row_data.as_ptr().add(i));
                    let d = vload_unaligned(batch_dst.as_ptr().add(i));
                    vstore_unaligned::<S, _>(batch_dst.as_mut_ptr().add(i), d + s);
                }
                i += lanes;
            }

            for j in simd_len..row_len {
                batch_dst[j] += row_data[j];
            }
        }
    }
}

// ============================================================================
// Row-wise sum (last-dim reduction)
// ============================================================================

/// Sum each row, storing results in output slice.
/// Used for last-dim reductions.
#[inline]
pub fn sum_rows_f32(src: &[f32], dst: &mut [f32], num_rows: usize, row_len: usize) {
    debug_assert_eq!(dst.len(), num_rows, "dst length must equal num_rows");
    debug_assert!(
        src.len() >= num_rows * row_len,
        "src too short: need {} elements, got {}",
        num_rows * row_len,
        src.len()
    );
    for (row, dst_val) in dst.iter_mut().enumerate() {
        let row_start = row * row_len;
        let row_data = &src[row_start..row_start + row_len];
        *dst_val = macerator_sum(row_data);
    }
}

// ============================================================================
// Max/Min reduction
// ============================================================================

/// Find the maximum element in a f32 slice using SIMD.
#[inline]
pub fn max_f32(data: &[f32]) -> f32 {
    macerator_max(data, f32::NEG_INFINITY)
}

/// Find the minimum element in a f32 slice using SIMD.
#[inline]
pub fn min_f32(data: &[f32]) -> f32 {
    macerator_min(data, f32::INFINITY)
}

#[macerator::with_simd]
fn macerator_max<S: Simd, F: VOrd + ReduceMax + PartialOrd>(mut xs: &[F], init: F) -> F {
    let lanes = F::lanes::<S>();
    let mut acc = init.splat::<S>();

    while xs.len() >= lanes {
        let v = unsafe { vload_unaligned(xs.as_ptr()) };
        acc = acc.max(v);
        xs = &xs[lanes..];
    }

    let mut result = acc.reduce_max();
    for &x in xs {
        if x > result {
            result = x;
        }
    }
    result
}

#[macerator::with_simd]
fn macerator_min<S: Simd, F: VOrd + ReduceMin + PartialOrd>(mut xs: &[F], init: F) -> F {
    let lanes = F::lanes::<S>();
    let mut acc = init.splat::<S>();

    while xs.len() >= lanes {
        let v = unsafe { vload_unaligned(xs.as_ptr()) };
        acc = acc.min(v);
        xs = &xs[lanes..];
    }

    let mut result = acc.reduce_min();
    for &x in xs {
        if x < result {
            result = x;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_f32() {
        let data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let expected: f32 = data.iter().sum();
        let result = sum_f32(&data);
        assert!((result - expected).abs() < 0.01);
    }

    #[test]
    fn test_sum_f32_empty() {
        let data: Vec<f32> = vec![];
        assert_eq!(sum_f32(&data), 0.0);
    }

    #[test]
    fn test_sum_f32_small() {
        let data = vec![1.0, 2.0, 3.0];
        assert_eq!(sum_f32(&data), 6.0);
    }

    #[test]
    fn test_scatter_add_f32() {
        // Simulate reducing [3, 4] along dim=0 -> [1, 4]
        let src = vec![
            1.0, 2.0, 3.0, 4.0, // row 0
            5.0, 6.0, 7.0, 8.0, // row 1
            9.0, 10.0, 11.0, 12.0, // row 2
        ];
        let mut dst = vec![0.0; 4];

        scatter_add_f32(&src, &mut dst, 3, 4, 4);

        assert_eq!(dst, vec![15.0, 18.0, 21.0, 24.0]);
    }

    #[test]
    fn test_sum_rows_f32() {
        // Simulate reducing [3, 4] along dim=1 -> [3, 1]
        let src = vec![
            1.0, 2.0, 3.0, 4.0, // row 0 -> 10
            5.0, 6.0, 7.0, 8.0, // row 1 -> 26
            9.0, 10.0, 11.0, 12.0, // row 2 -> 42
        ];
        let mut dst = vec![0.0; 3];

        sum_rows_f32(&src, &mut dst, 3, 4);

        assert_eq!(dst, vec![10.0, 26.0, 42.0]);
    }

    #[test]
    fn test_max_f32() {
        let data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        assert_eq!(max_f32(&data), 999.0);
    }

    #[test]
    fn test_max_f32_small() {
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        assert_eq!(max_f32(&data), 5.0);
    }

    #[test]
    fn test_max_f32_negative() {
        let data = vec![-3.0, -1.0, -4.0, -1.0, -5.0];
        assert_eq!(max_f32(&data), -1.0);
    }

    #[test]
    fn test_min_f32() {
        let data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        assert_eq!(min_f32(&data), 0.0);
    }

    #[test]
    fn test_min_f32_small() {
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        assert_eq!(min_f32(&data), 1.0);
    }

    #[test]
    fn test_min_f32_negative() {
        let data = vec![-3.0, -1.0, -4.0, -1.0, -5.0];
        assert_eq!(min_f32(&data), -5.0);
    }
}
