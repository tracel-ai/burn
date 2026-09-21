//! Transposed convolution operations using GEMM + col2im approach.
//!
//! All transposed convolutions (1D, 2D, 3D) use a unified 3D implementation:
//! - conv_transpose1d: adds two size-1 dimensions, calls conv_transpose3d, squeezes output
//! - conv_transpose2d: adds one size-1 dimension, calls conv_transpose3d, squeezes output
//! - conv_transpose3d: native implementation
//!
//! For each (batch, group), computes: columns = W_g^T @ X_g, then scatters
//! the columns matrix into the output via col2im. The GEMM handles the heavy
//! channel-reduction multiply-adds; col2im is a lightweight spatial scatter.
//!
//! Supported dtypes: f32, f64, f16 (native gemm), bf16 (via f32 conversion)

use alloc::vec;
use burn_backend::DType;
use burn_backend::ops::ConvTransposeOptions;
use burn_backend::ops::conv::calculate_conv_transpose_output_size;
use burn_std::{Bytes, Shape, f16};

use crate::{FlexTensor, Layout};

use super::conv_common::{add_bias, squeeze_3d_to_1d, squeeze_3d_to_2d};

/// Generates a conv_transpose3d typed function.
macro_rules! conv_transpose3d_typed {
    ($fn_name:ident, $T:ty, $dtype:expr, $zero:expr, $gemm_fn:ident) => {
        pub fn $fn_name(
            x: FlexTensor,
            weight: FlexTensor,
            bias: Option<FlexTensor>,
            options: &ConvTransposeOptions<3>,
        ) -> FlexTensor {
            const COLUMNS_BYTES: usize = 8 * 1024 * 1024;
            let columns_bytes = weight.layout().shape()[1..]
                .iter()
                .chain(&x.layout().shape()[2..])
                .try_fold(core::mem::size_of::<$T>(), |size, &dim| {
                    size.checked_mul(dim)
                });
            // Keep the original loop for small buffers, without tile bounds.
            if columns_bytes.is_some_and(|bytes| bytes <= COLUMNS_BYTES) {
                conv_transpose3d_impl::<$T, COLUMNS_BYTES, false>(
                    x, weight, bias, options, $dtype, $zero, $gemm_fn,
                )
            } else {
                conv_transpose3d_impl::<$T, COLUMNS_BYTES, true>(
                    x, weight, bias, options, $dtype, $zero, $gemm_fn,
                )
            }
        }
    };
}

// ============================================================================
// Conv Transpose 1d - delegates to conv_transpose3d
// ============================================================================

conv_nd_via_3d!(
    conv_transpose1d_f32,
    conv_transpose3d_f32,
    expand_transpose_1d_to_3d,
    squeeze_3d_to_1d,
    1,
    ConvTransposeOptions
);
conv_nd_via_3d!(
    conv_transpose1d_f64,
    conv_transpose3d_f64,
    expand_transpose_1d_to_3d,
    squeeze_3d_to_1d,
    1,
    ConvTransposeOptions
);
conv_nd_via_3d!(
    conv_transpose1d_f16,
    conv_transpose3d_f16,
    expand_transpose_1d_to_3d,
    squeeze_3d_to_1d,
    1,
    ConvTransposeOptions
);
bf16_via_f32!(
    conv_transpose1d_bf16,
    conv_transpose1d_f32,
    1,
    ConvTransposeOptions
);

fn expand_transpose_1d_to_3d(
    x: &FlexTensor,
    weight: &FlexTensor,
    options: &ConvTransposeOptions<1>,
) -> (FlexTensor, FlexTensor, ConvTransposeOptions<3>) {
    // x: [N, C_in, L] -> [N, C_in, 1, 1, L]
    let x_shape = x.layout().shape();
    let x_3d = x.reshape(Shape::from(vec![x_shape[0], x_shape[1], 1, 1, x_shape[2]]));

    // weight: [C_in, C_out, K] -> [C_in, C_out, 1, 1, K]
    let w_shape = weight.layout().shape();
    let weight_3d = weight.reshape(Shape::from(vec![w_shape[0], w_shape[1], 1, 1, w_shape[2]]));

    let options_3d = ConvTransposeOptions::new(
        [1, 1, options.stride[0]],
        [0, 0, options.padding[0]],
        [0, 0, options.padding_out[0]],
        [1, 1, options.dilation[0]],
        options.groups,
    );

    (x_3d, weight_3d, options_3d)
}

// ============================================================================
// Conv Transpose 2d - delegates to conv_transpose3d
// ============================================================================

conv_nd_via_3d!(
    conv_transpose2d_f32,
    conv_transpose3d_f32,
    expand_transpose_2d_to_3d,
    squeeze_3d_to_2d,
    2,
    ConvTransposeOptions
);
conv_nd_via_3d!(
    conv_transpose2d_f64,
    conv_transpose3d_f64,
    expand_transpose_2d_to_3d,
    squeeze_3d_to_2d,
    2,
    ConvTransposeOptions
);
conv_nd_via_3d!(
    conv_transpose2d_f16,
    conv_transpose3d_f16,
    expand_transpose_2d_to_3d,
    squeeze_3d_to_2d,
    2,
    ConvTransposeOptions
);
bf16_via_f32!(
    conv_transpose2d_bf16,
    conv_transpose2d_f32,
    2,
    ConvTransposeOptions
);

fn expand_transpose_2d_to_3d(
    x: &FlexTensor,
    weight: &FlexTensor,
    options: &ConvTransposeOptions<2>,
) -> (FlexTensor, FlexTensor, ConvTransposeOptions<3>) {
    // x: [N, C_in, H, W] -> [N, C_in, 1, H, W]
    let x_shape = x.layout().shape();
    let x_3d = x.reshape(Shape::from(vec![
        x_shape[0], x_shape[1], 1, x_shape[2], x_shape[3],
    ]));

    // weight: [C_in, C_out, Kh, Kw] -> [C_in, C_out, 1, Kh, Kw]
    let w_shape = weight.layout().shape();
    let weight_3d = weight.reshape(Shape::from(vec![
        w_shape[0], w_shape[1], 1, w_shape[2], w_shape[3],
    ]));

    let options_3d = ConvTransposeOptions::new(
        [1, options.stride[0], options.stride[1]],
        [0, options.padding[0], options.padding[1]],
        [0, options.padding_out[0], options.padding_out[1]],
        [1, options.dilation[0], options.dilation[1]],
        options.groups,
    );

    (x_3d, weight_3d, options_3d)
}

// ============================================================================
// Conv Transpose 3d - core implementation
// ============================================================================

conv_transpose3d_typed!(
    conv_transpose3d_f32,
    f32,
    DType::F32,
    0.0f32,
    conv_transpose_gemm_f32
);
conv_transpose3d_typed!(
    conv_transpose3d_f64,
    f64,
    DType::F64,
    0.0f64,
    conv_transpose_gemm_f64
);
conv_transpose3d_typed!(
    conv_transpose3d_f16,
    f16,
    DType::F16,
    f16::from_f32(0.0),
    conv_transpose_gemm_f16
);
bf16_via_f32!(
    conv_transpose3d_bf16,
    conv_transpose3d_f32,
    3,
    ConvTransposeOptions
);

/// GEMM for conv_transpose: `c = a^T @ b`, with a row stride for each input tile.
type ConvTransposeGemmFn<T> = fn(&mut [T], &[T], &[T], usize, usize, usize, usize);

/// Compute the half-open range `[in_start, in_end)` of input spatial positions `i`
/// for which the corresponding output index `i * stride + k * dilation - pad`
/// lies inside `[0, out_size)`.
#[inline]
fn valid_in_range(
    k: usize,
    dilation: usize,
    pad: usize,
    stride: usize,
    in_size: usize,
    out_size: usize,
) -> (usize, usize) {
    debug_assert!(stride >= 1, "stride must be >= 1");
    let offset = k * dilation;
    let in_start = if offset >= pad {
        0
    } else {
        (pad - offset).div_ceil(stride)
    };
    let threshold = out_size + pad;
    let in_end = if offset >= threshold {
        0
    } else {
        (threshold - offset).div_ceil(stride)
    };
    let in_end = in_end.min(in_size);
    let in_start = in_start.min(in_end);
    (in_start, in_end)
}

/// 3D transposed convolution via GEMM + col2im.
#[allow(clippy::too_many_arguments)]
fn conv_transpose3d_impl<
    T: bytemuck::Pod + Clone + Copy + Send + Sync + burn_backend::Element + burn_backend::ElementAdd,
    const COLUMNS_BYTES: usize,
    const TILED: bool,
>(
    x: FlexTensor,
    weight: FlexTensor,
    bias: Option<FlexTensor>,
    options: &ConvTransposeOptions<3>,
    dtype: DType,
    zero: T,
    gemm_fn: ConvTransposeGemmFn<T>,
) -> FlexTensor {
    let x = x.to_contiguous();
    let weight = weight.to_contiguous();

    let x_shape = x.layout().shape();
    let w_shape = weight.layout().shape();

    let batch_size = x_shape[0];
    let in_channels = x_shape[1];
    let in_d = x_shape[2];
    let in_h = x_shape[3];
    let in_w = x_shape[4];

    // Weight shape for transpose: [in_channels, out_channels_per_group, kd, kh, kw]
    let out_channels_per_group = w_shape[1];
    let kernel_d = w_shape[2];
    let kernel_h = w_shape[3];
    let kernel_w = w_shape[4];

    let [stride_d, stride_h, stride_w] = options.stride;
    let [pad_d, pad_h, pad_w] = options.padding;
    let [pad_out_d, pad_out_h, pad_out_w] = options.padding_out;
    let [dilation_d, dilation_h, dilation_w] = options.dilation;
    let groups = options.groups;

    let out_channels = out_channels_per_group * groups;
    let in_channels_per_group = in_channels / groups;

    let out_d = calculate_conv_transpose_output_size(
        kernel_d, stride_d, pad_d, pad_out_d, dilation_d, in_d,
    );
    let out_h = calculate_conv_transpose_output_size(
        kernel_h, stride_h, pad_h, pad_out_h, dilation_h, in_h,
    );
    let out_w = calculate_conv_transpose_output_size(
        kernel_w, stride_w, pad_w, pad_out_w, dilation_w, in_w,
    );

    let x_data: &[T] = x.storage();
    let w_data: &[T] = weight.storage();

    let k_spatial = [kernel_d, kernel_h, kernel_w]
        .iter()
        .try_fold(1usize, |acc, &x| acc.checked_mul(x))
        .expect("conv_transpose: kernel dimensions would overflow");
    let in_spatial = [in_d, in_h, in_w]
        .iter()
        .try_fold(1usize, |acc, &x| acc.checked_mul(x))
        .expect("conv_transpose: input spatial dimensions would overflow");
    let out_spatial = out_d * out_h * out_w;
    let col_ch = out_channels_per_group
        .checked_mul(k_spatial)
        .expect("conv_transpose: columns dimensions would overflow");
    // Limit the column buffer per batch/group. One input position is the minimum
    // tile, even when its columns alone exceed the byte limit.
    let tile_size = if TILED {
        (COLUMNS_BYTES / core::mem::size_of::<T>() / col_ch.max(1))
            .max(1)
            .min(in_spatial.max(1))
    } else {
        in_spatial.max(1)
    };
    let columns_len = col_ch
        .checked_mul(in_spatial.min(tile_size))
        .expect("conv_transpose: columns buffer size would overflow");

    let output_size = [batch_size, out_channels, out_d, out_h, out_w]
        .iter()
        .try_fold(1usize, |acc, &x| acc.checked_mul(x))
        .expect("conv_transpose: output dimensions would overflow");
    if output_size == 0 {
        // Empty spatial output would otherwise reach chunks_mut(0).
        return FlexTensor::empty(
            Shape::from(vec![batch_size, out_channels, out_d, out_h, out_w]),
            dtype,
        );
    }
    let mut output = vec![zero; output_size];

    let group_chunk_len = out_channels_per_group * out_spatial;

    let process_group = |b: usize, g: usize, group_output: &mut [T], columns: &mut [T]| {
        let ic_start = g * in_channels_per_group;
        let x_offset = b * in_channels * in_spatial + ic_start * in_spatial;
        let w_offset = ic_start * out_channels_per_group * k_spatial;

        let x_group = &x_data[x_offset..x_offset + in_channels_per_group * in_spatial];
        let w_group = &w_data[w_offset..w_offset + in_channels_per_group * col_ch];

        if in_channels_per_group == 0 {
            return;
        }
        let mut process_tile = |tile_start: usize, tile_len: usize| {
            let tile_end = tile_start + tile_len;
            let columns = &mut columns[..col_ch * tile_len];
            gemm_fn(
                columns,
                w_group,
                &x_group[tile_start..],
                col_ch,
                in_channels_per_group,
                tile_len,
                in_spatial,
            );

            // Add this tile to the output for this (batch, group).
            for oc in 0..out_channels_per_group {
                let out_ch_base = oc * out_spatial;
                let oc_col_base = oc * k_spatial;

                for kd in 0..kernel_d {
                    let (id_start, id_end) =
                        valid_in_range(kd, dilation_d, pad_d, stride_d, in_d, out_d);
                    let (id_start, id_end) = if TILED {
                        (
                            id_start.max(tile_start / (in_h * in_w)),
                            id_end.min(tile_end.div_ceil(in_h * in_w)),
                        )
                    } else {
                        (id_start, id_end)
                    };
                    if id_start >= id_end {
                        continue;
                    }
                    for kh in 0..kernel_h {
                        let (ih_start, ih_end) =
                            valid_in_range(kh, dilation_h, pad_h, stride_h, in_h, out_h);
                        if ih_start >= ih_end {
                            continue;
                        }
                        for kw in 0..kernel_w {
                            let (iw_start, iw_end) =
                                valid_in_range(kw, dilation_w, pad_w, stride_w, in_w, out_w);
                            if iw_start >= iw_end {
                                continue;
                            }
                            let k_idx = kd * kernel_h * kernel_w + kh * kernel_w + kw;
                            let col_base = (oc_col_base + k_idx) * tile_len;

                            for id in id_start..id_end {
                                let od = id * stride_d + kd * dilation_d - pad_d;
                                let in_d_base = id * in_h * in_w;
                                let out_d_base = od * out_h * out_w;
                                let (ih_start, ih_end) = if TILED {
                                    (
                                        ih_start.max(tile_start.saturating_sub(in_d_base) / in_w),
                                        ih_end
                                            .min(tile_end.saturating_sub(in_d_base).div_ceil(in_w)),
                                    )
                                } else {
                                    (ih_start, ih_end)
                                };

                                for ih in ih_start..ih_end {
                                    let oh = ih * stride_h + kh * dilation_h - pad_h;
                                    let in_h_base = in_d_base + ih * in_w;
                                    let out_h_base = out_d_base + oh * out_w;
                                    let (iw_start, iw_end) = if TILED {
                                        (
                                            iw_start.max(tile_start.saturating_sub(in_h_base)),
                                            iw_end.min(tile_end.saturating_sub(in_h_base)),
                                        )
                                    } else {
                                        (iw_start, iw_end)
                                    };
                                    for iw in iw_start..iw_end {
                                        let ow = iw * stride_w + kw * dilation_w - pad_w;
                                        let s = in_h_base + iw - tile_start;
                                        let val = columns[col_base + s];
                                        let out_idx = out_ch_base + out_h_base + ow;
                                        group_output[out_idx] = T::add(group_output[out_idx], val);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        };
        if TILED {
            // For a fixed output, increasing kernel indices read decreasing input
            // positions. Visit tiles in reverse to keep the col2im addition order.
            for tile_start in (0..in_spatial).step_by(tile_size).rev() {
                process_tile(tile_start, tile_size.min(in_spatial - tile_start));
            }
        } else {
            process_tile(0, in_spatial);
        }
    };

    #[cfg(feature = "rayon")]
    let total_groups = batch_size * groups;
    #[cfg(feature = "rayon")]
    if total_groups > 1
        && (total_groups * out_channels_per_group * out_spatial) >= super::PARALLEL_THRESHOLD
    {
        use rayon::prelude::*;
        output
            .par_chunks_mut(group_chunk_len)
            .enumerate()
            .for_each(|(bg, group_output)| {
                let b = bg / groups;
                let g = bg % groups;
                let mut columns = vec![zero; columns_len];
                process_group(b, g, group_output, &mut columns);
            });
    } else {
        let mut columns = vec![zero; columns_len];
        for (bg, group_output) in output.chunks_mut(group_chunk_len).enumerate() {
            let b = bg / groups;
            let g = bg % groups;
            process_group(b, g, group_output, &mut columns);
        }
    }
    #[cfg(not(feature = "rayon"))]
    {
        let mut columns = vec![zero; columns_len];
        for (bg, group_output) in output.chunks_mut(group_chunk_len).enumerate() {
            let b = bg / groups;
            let g = bg % groups;
            process_group(b, g, group_output, &mut columns);
        }
    }

    // Add bias if present
    if let Some(bias) = bias {
        let bias = bias.to_contiguous();
        let bias_data: &[T] = bias.storage();
        add_bias(
            &mut output,
            bias_data,
            batch_size,
            out_channels,
            out_spatial,
        );
    }

    let out_shape = Shape::from(vec![batch_size, out_channels, out_d, out_h, out_w]);
    FlexTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(out_shape),
        dtype,
    )
}

// ============================================================================
// gemm for conv_transpose: C[m,n] = A[k,m]^T @ B[k,n]
// ============================================================================

macro_rules! conv_transpose_gemm_typed {
    ($fn_name:ident, $T:ty, $zero:expr, $one:expr) => {
        fn $fn_name(
            c: &mut [$T],
            a: &[$T],
            b: &[$T],
            m: usize,
            k: usize,
            n: usize,
            b_stride: usize,
        ) {
            debug_assert_eq!(c.len(), m * n);
            debug_assert_eq!(a.len(), k * m);
            debug_assert!(k == 0 || b.len() >= (k - 1) * b_stride + n);
            #[cfg(feature = "rayon")]
            let parallelism = if m * n * k >= 192 * 192 * 192 {
                gemm::Parallelism::Rayon(0)
            } else {
                gemm::Parallelism::None
            };
            #[cfg(not(feature = "rayon"))]
            let parallelism = gemm::Parallelism::None;
            unsafe {
                gemm::gemm(
                    m,
                    n,
                    k,
                    c.as_mut_ptr(),
                    1,          // dst_cs
                    n as isize, // dst_rs
                    false,
                    a.as_ptr(),
                    m as isize, // lhs_cs: A^T column stride = row length of A
                    1,          // lhs_rs: A^T row stride = 1
                    b.as_ptr(),
                    1,                 // rhs_cs
                    b_stride as isize, // rhs_rs
                    $zero,
                    $one,
                    false,
                    false,
                    false,
                    parallelism,
                );
            }
        }
    };
}

conv_transpose_gemm_typed!(conv_transpose_gemm_f32, f32, 0.0f32, 1.0f32);
conv_transpose_gemm_typed!(conv_transpose_gemm_f64, f64, 0.0f64, 1.0f64);
conv_transpose_gemm_typed!(
    conv_transpose_gemm_f16,
    f16,
    f16::from_f32(0.0),
    f16::from_f32(1.0)
);

// ============================================================================
// Tests
// ============================================================================

// Tests kept here exercise flex-specific dtype storage paths (f64/f16)
// through the conv_transpose dispatch. Plain conv_transpose shape/stride/
// padding/groups/dilation/multichannel/batch correctness is covered at
// the public API level by crates/burn-backend-tests/tests/tensor/float/
// module/conv_transpose{1,2,3}d.rs, so those tests were removed from here.
// When adding new tests, keep them here only if they probe flex dtype
// dispatch or a flex-internal fast path; otherwise add them there.
#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::TensorData;

    fn check_tiles<const BYTES: usize>(spatial: [usize; 3], options: ConvTransposeOptions<3>) {
        let [d, h, w] = spatial;
        let x_shape = [2, 4, d, h, w];
        let w_shape = [4, 3, 3, 2, 4];
        let x_values: Vec<f32> = (0..x_shape.iter().product())
            .map(|i| ((i * 17 % 31) as f32 - 15.0) / 8.0)
            .collect();
        let w_values: Vec<f32> = (0..w_shape.iter().product())
            .map(|i| ((i * 7 % 19) as f32 - 9.0) / 16.0)
            .collect();
        let bias_values: Vec<f32> = (0..3 * options.groups).map(|i| i as f32 / 4.0).collect();
        let x = FlexTensor::from_data(TensorData::new(x_values.clone(), x_shape));
        let weight = FlexTensor::from_data(TensorData::new(w_values.clone(), w_shape));
        let bias =
            FlexTensor::from_data(TensorData::new(bias_values.clone(), [3 * options.groups]));
        let result = conv_transpose3d_impl::<f32, BYTES, true>(
            x,
            weight,
            Some(bias),
            &options,
            DType::F32,
            0.0,
            conv_transpose_gemm_f32,
        );
        let shape = result.layout().shape();
        let [od, oh, ow] = [shape[2], shape[3], shape[4]];
        let output = result.storage::<f32>();

        // Gather each output from the input directly, without a column matrix.
        for b in 0..2 {
            for (oc, &bias_value) in bias_values.iter().enumerate() {
                let group = oc / 3;
                for z in 0..od {
                    for y in 0..oh {
                        for x in 0..ow {
                            let mut expected = 0.0;
                            for kz in 0..3 {
                                for ky in 0..2 {
                                    for kx in 0..4 {
                                        let mut input = [0; 3];
                                        let mut valid = true;
                                        for (axis, (o, k)) in
                                            [z, y, x].into_iter().zip([kz, ky, kx]).enumerate()
                                        {
                                            let i = o as isize + options.padding[axis] as isize
                                                - (k * options.dilation[axis]) as isize;
                                            let stride = options.stride[axis] as isize;
                                            if i < 0
                                                || i % stride != 0
                                                || i / stride >= spatial[axis] as isize
                                            {
                                                valid = false;
                                                break;
                                            }
                                            input[axis] = (i / stride) as usize;
                                        }
                                        if !valid {
                                            continue;
                                        }
                                        let mut dot = 0.0;
                                        for ci in group * (4 / options.groups)
                                            ..(group + 1) * (4 / options.groups)
                                        {
                                            let xi = (((b * 4 + ci) * d + input[0]) * h + input[1])
                                                * w
                                                + input[2];
                                            let wi =
                                                (((ci * 3 + oc % 3) * 3 + kz) * 2 + ky) * 4 + kx;
                                            dot += x_values[xi] * w_values[wi];
                                        }
                                        expected += dot;
                                    }
                                }
                            }
                            expected += bias_value;
                            let index = (((b * shape[1] + oc) * od + z) * oh + y) * ow + x;
                            assert_eq!(
                                output[index], expected,
                                "output {index}, tile limit {BYTES}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_conv_transpose_tiles_cross_rows_and_planes() {
        for groups in [1, 2] {
            for spatial in [[1, 1, 11], [1, 5, 7], [3, 4, 5]] {
                let options =
                    ConvTransposeOptions::new([2, 1, 3], [1, 1, 2], [1, 0, 2], [2, 2, 1], groups);
                check_tiles::<{ 7 * 3 * 3 * 2 * 4 * 4 }>(spatial, options.clone());
                // A single position can exceed the byte limit.
                check_tiles::<1>(spatial, options);
            }
        }
    }

    macro_rules! check_addition_order {
        ($name:ident, $ty:ty, $dtype:expr, $gemm:ident, $values:expr, $one:expr, $zero:expr) => {
            #[test]
            fn $name() {
                let values: &[$ty] = &$values;
                let x = FlexTensor::from_data(TensorData::new(
                    (0..180)
                        .map(|i| values[i % values.len()])
                        .collect::<Vec<_>>(),
                    [1, 3, 3, 4, 5],
                ));
                let one: $ty = $one;
                let weight = FlexTensor::from_data(TensorData::new(vec![one; 81], [3, 1, 3, 3, 3]));
                let options = ConvTransposeOptions::new([1; 3], [1; 3], [0; 3], [1; 3], 1);
                let full = conv_transpose3d_impl::<$ty, { usize::MAX }, false>(
                    x.clone(),
                    weight.clone(),
                    None,
                    &options,
                    $dtype,
                    $zero,
                    $gemm,
                );
                let tiled =
                    conv_transpose3d_impl::<$ty, { 7 * 27 * core::mem::size_of::<$ty>() }, true>(
                        x, weight, None, &options, $dtype, $zero, $gemm,
                    );
                assert_eq!(full.storage::<$ty>(), tiled.storage::<$ty>());
            }
        };
    }

    // Cancellation makes changes to the col2im addition order visible.
    check_addition_order!(
        test_tiles_f32_addition_order,
        f32,
        DType::F32,
        conv_transpose_gemm_f32,
        [1e20, 1.0, -1e20, 3.0, 0.5],
        1.0,
        0.0
    );
    check_addition_order!(
        test_tiles_f64_addition_order,
        f64,
        DType::F64,
        conv_transpose_gemm_f64,
        [1e100, 1.0, -1e100, 3.0, 0.5],
        1.0,
        0.0
    );
    check_addition_order!(
        test_tiles_f16_addition_order,
        f16,
        DType::F16,
        conv_transpose_gemm_f16,
        [2048.0, 1.0, -2048.0, 3.0, 0.5].map(f16::from_f32),
        f16::ONE,
        f16::ZERO
    );

    #[test]
    fn test_conv_transpose_tiles_empty_channels() {
        let x = FlexTensor::from_data(TensorData::new(Vec::<f32>::new(), [1, 0, 2, 3, 4]));
        let weight = FlexTensor::from_data(TensorData::new(Vec::<f32>::new(), [0, 1, 2, 2, 2]));
        let bias = FlexTensor::from_data(TensorData::new(vec![2.0f32], [1]));
        let options = ConvTransposeOptions::new([1; 3], [0; 3], [0; 3], [1; 3], 1);
        let result = conv_transpose3d_impl::<f32, 1, true>(
            x,
            weight,
            Some(bias),
            &options,
            DType::F32,
            0.0,
            conv_transpose_gemm_f32,
        );
        assert_eq!(result.storage::<f32>(), vec![2.0; 3 * 4 * 5]);
    }

    #[test]
    fn test_conv_transpose_empty_output_group() {
        // Padding removes the entire spatial output. The group copier must
        // return before constructing chunks with a zero length.
        let x = FlexTensor::from_data(TensorData::new(vec![1.0f32, 2.0], [1, 1, 2]));
        let weight = FlexTensor::from_data(TensorData::new(vec![1.0f32], [1, 1, 1]));
        let options = ConvTransposeOptions::new([1], [1], [0], [1], 1);
        let result = conv_transpose1d_f32(x, weight, None, &options);
        assert_eq!(result.layout().shape().to_vec(), vec![1, 1, 0]);
        assert_eq!(result.dtype(), DType::F32);
        assert!(result.into_data().bytes.is_empty());
    }

    #[test]
    fn test_conv_transpose2d_f64() {
        let x = FlexTensor::from_data(TensorData::new(
            vec![1.0f64, 2.0, 3.0, 4.0],
            vec![1, 1, 2, 2],
        ));
        let w = FlexTensor::from_data(TensorData::new(vec![1.0f64; 4], vec![1, 1, 2, 2]));
        let opts = ConvTransposeOptions::new([1, 1], [0, 0], [0, 0], [1, 1], 1);
        let result = conv_transpose2d_f64(x, w, None, &opts);
        let out: Vec<f64> = result.into_data().try_into_vec().unwrap();
        assert_eq!(out, vec![1.0, 3.0, 2.0, 4.0, 10.0, 6.0, 3.0, 7.0, 4.0]);
    }

    #[test]
    fn test_conv_transpose2d_f16() {
        let x_data: Vec<f16> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .map(|&v| f16::from_f32(v))
            .collect();
        let w_data: Vec<f16> = [1.0f32; 4].iter().map(|&v| f16::from_f32(v)).collect();
        let x = FlexTensor::from_data(TensorData::new(x_data, vec![1, 1, 2, 2]));
        let w = FlexTensor::from_data(TensorData::new(w_data, vec![1, 1, 2, 2]));
        let opts = ConvTransposeOptions::new([1, 1], [0, 0], [0, 0], [1, 1], 1);
        let result = conv_transpose2d_f16(x, w, None, &opts);
        let out: Vec<f16> = result.into_data().try_into_vec().unwrap();
        let expected = [1.0f32, 3.0, 2.0, 4.0, 10.0, 6.0, 3.0, 7.0, 4.0];
        for (a, e) in out.iter().zip(expected.iter()) {
            assert!((a.to_f32() - e).abs() < 0.1);
        }
    }
}
