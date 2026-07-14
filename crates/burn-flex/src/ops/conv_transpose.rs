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
    ($fn_name:ident, $T:ty, $dtype:expr, $zero:expr, $gemm_fn:ident, $add_fn:expr) => {
        pub fn $fn_name(
            x: FlexTensor,
            weight: FlexTensor,
            bias: Option<FlexTensor>,
            options: &ConvTransposeOptions<3>,
        ) -> FlexTensor {
            conv_transpose3d_impl::<$T>(x, weight, bias, options, $dtype, $zero, $gemm_fn, $add_fn)
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
    conv_transpose_gemm_f32,
    |a, b| a + b
);
conv_transpose3d_typed!(
    conv_transpose3d_f64,
    f64,
    DType::F64,
    0.0f64,
    conv_transpose_gemm_f64,
    |a, b| a + b
);
conv_transpose3d_typed!(
    conv_transpose3d_f16,
    f16,
    DType::F16,
    f16::from_f32(0.0),
    conv_transpose_gemm_f16,
    |a: f16, b: f16| f16::from_f32(a.to_f32() + b.to_f32())
);
bf16_via_f32!(
    conv_transpose3d_bf16,
    conv_transpose3d_f32,
    3,
    ConvTransposeOptions
);

/// GEMM for conv_transpose: writes `c = a^T @ b` where a is [k,m] and b is [k,n].
type ConvTransposeGemmFn<T> = fn(&mut [T], &[T], &[T], usize, usize, usize);

/// 3D transposed convolution via GEMM + col2im.
#[allow(clippy::too_many_arguments)]
fn conv_transpose3d_impl<T: bytemuck::Pod + Clone + Copy + Send + Sync + burn_backend::Element>(
    x: FlexTensor,
    weight: FlexTensor,
    bias: Option<FlexTensor>,
    options: &ConvTransposeOptions<3>,
    dtype: DType,
    zero: T,
    gemm_fn: ConvTransposeGemmFn<T>,
    add_fn: fn(T, T) -> T,
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
    let columns_len = col_ch
        .checked_mul(in_spatial)
        .expect("conv_transpose: columns buffer size would overflow");

    let output_size = [batch_size, out_channels, out_d, out_h, out_w]
        .iter()
        .try_fold(1usize, |acc, &x| acc.checked_mul(x))
        .expect("conv_transpose: output dimensions would overflow");
    let mut output = vec![zero; output_size];

    // Reuse columns buffer across (batch, group) iterations; GEMM overwrites it fully.
    let mut columns = vec![zero; columns_len];

    for b in 0..batch_size {
        for g in 0..groups {
            let ic_start = g * in_channels_per_group;
            let oc_start = g * out_channels_per_group;

            let x_offset = b * in_channels * in_spatial + ic_start * in_spatial;
            let w_offset = ic_start * out_channels_per_group * k_spatial;

            let x_group = &x_data[x_offset..x_offset + in_channels_per_group * in_spatial];
            let w_group = &w_data[w_offset..w_offset + in_channels_per_group * col_ch];

            gemm_fn(
                &mut columns,
                w_group,
                x_group,
                col_ch,
                in_channels_per_group,
                in_spatial,
            );

            // col2im: scatter columns into output for this (batch, group)
            let out_base = b * out_channels * out_spatial;
            for oc in 0..out_channels_per_group {
                let out_ch_base = out_base + (oc_start + oc) * out_spatial;
                let oc_col_base = oc * k_spatial;

                for kd in 0..kernel_d {
                    for kh in 0..kernel_h {
                        for kw in 0..kernel_w {
                            let k_idx = kd * kernel_h * kernel_w + kh * kernel_w + kw;
                            let col_base = (oc_col_base + k_idx) * in_spatial;

                            for id in 0..in_d {
                                let od_raw = id * stride_d + kd * dilation_d;
                                if od_raw < pad_d {
                                    continue;
                                }
                                let od = od_raw - pad_d;
                                if od >= out_d {
                                    continue;
                                }

                                for ih in 0..in_h {
                                    let oh_raw = ih * stride_h + kh * dilation_h;
                                    if oh_raw < pad_h {
                                        continue;
                                    }
                                    let oh = oh_raw - pad_h;
                                    if oh >= out_h {
                                        continue;
                                    }

                                    for iw in 0..in_w {
                                        let ow_raw = iw * stride_w + kw * dilation_w;
                                        if ow_raw < pad_w {
                                            continue;
                                        }
                                        let ow = ow_raw - pad_w;
                                        if ow >= out_w {
                                            continue;
                                        }

                                        let s = id * in_h * in_w + ih * in_w + iw;
                                        let val = columns[col_base + s];
                                        let out_idx =
                                            out_ch_base + od * out_h * out_w + oh * out_w + ow;
                                        output[out_idx] = add_fn(output[out_idx], val);
                                    }
                                }
                            }
                        }
                    }
                }
            }
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
            add_fn,
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
        fn $fn_name(c: &mut [$T], a: &[$T], b: &[$T], m: usize, k: usize, n: usize) {
            debug_assert_eq!(c.len(), m * n);
            debug_assert_eq!(a.len(), k * m);
            debug_assert_eq!(b.len(), k * n);
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
                    1,          // rhs_cs
                    n as isize, // rhs_rs
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

    #[test]
    fn test_conv_transpose2d_f64() {
        let x = FlexTensor::from_data(TensorData::new(
            vec![1.0f64, 2.0, 3.0, 4.0],
            vec![1, 1, 2, 2],
        ));
        let w = FlexTensor::from_data(TensorData::new(vec![1.0f64; 4], vec![1, 1, 2, 2]));
        let opts = ConvTransposeOptions::new([1, 1], [0, 0], [0, 0], [1, 1], 1);
        let result = conv_transpose2d_f64(x, w, None, &opts);
        let out: Vec<f64> = result.into_data().to_vec().unwrap();
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
        let out: Vec<f16> = result.into_data().to_vec().unwrap();
        let expected = [1.0f32, 3.0, 2.0, 4.0, 10.0, 6.0, 3.0, 7.0, 4.0];
        for (a, e) in out.iter().zip(expected.iter()) {
            assert!((a.to_f32() - e).abs() < 0.1);
        }
    }
}
