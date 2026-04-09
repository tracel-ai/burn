//! Forward convolution operations using tiled im2col + gemm approach.
//!
//! All convolutions (1D, 2D, 3D) use a unified 3D implementation:
//! - conv1d: adds two size-1 dimensions, calls conv3d, squeezes output
//! - conv2d: adds one size-1 dimension, calls conv3d, squeezes output
//! - conv3d: native implementation
//!
//! Optimizations:
//! - Tiled im2col: Process output in tiles for better cache usage and parallelism
//! - NHWC layout: Convert to channels-last for cache-friendly access
//! - Nested parallelism: Batch and tile dimensions run in parallel via rayon
//! - 1x1 fast path: Skip im2col for pointwise convolutions
//! - Direct conv path: For small-spatial 1D-like convolutions, decompose into
//!   per-kernel-position gemm calls on NCHW data, skipping NHWC conversion and im2col
//!
//! Supported dtypes: f32, f64, f16 (native gemm), bf16 (via f32 conversion)

use alloc::vec;
use alloc::vec::Vec;
use burn_backend::DType;
use burn_backend::ops::ConvOptions;
use burn_backend::ops::conv::calculate_conv_output_size;
use burn_std::{Bytes, Shape, f16};

use crate::{FlexTensor, Layout};

use super::conv_common::{add_bias, squeeze_3d_to_1d, squeeze_3d_to_2d};

// ============================================================================
// Macros for forward conv
// ============================================================================

/// Generates a conv3d_1x1 function that uses the optimized gemm fast path.
macro_rules! conv3d_1x1_typed {
    ($fn_name:ident, $T:ty, $dtype:expr, $zero:expr, $one:expr, $add_fn:expr) => {
        fn $fn_name(
            x: FlexTensor,
            weight: FlexTensor,
            bias: Option<FlexTensor>,
            options: &ConvOptions<3>,
        ) -> FlexTensor {
            conv3d_1x1_impl::<$T>(x, weight, bias, options, $dtype, $zero, $one, $add_fn)
        }
    };
}

/// Generates a conv3d typed function with 1x1 and direct fast-path checks.
macro_rules! conv3d_typed {
    ($fn_name:ident, $T:ty, $dtype:expr, $zero:expr, $gemm_fn:ident, $add_fn:expr, $fn_1x1:ident $(, $fn_direct:ident)?) => {
        pub fn $fn_name(
            x: FlexTensor,
            weight: FlexTensor,
            bias: Option<FlexTensor>,
            options: &ConvOptions<3>,
        ) -> FlexTensor {
            let w_shape = weight.layout().shape();
            if is_1x1_conv(w_shape[2], w_shape[3], w_shape[4], options) {
                return $fn_1x1(x, weight, bias, options);
            }
            $(
                let x_shape = x.layout().shape();
                if should_use_direct_conv(x_shape, w_shape, options) {
                    return $fn_direct(x, weight, bias, options);
                }
            )?
            conv3d_impl::<$T>(x, weight, bias, options, $dtype, $zero, $gemm_fn, $add_fn)
        }
    };
}

// ============================================================================
// Conv1d - delegates to conv3d
// ============================================================================

conv_nd_via_3d!(
    conv1d_f32,
    conv3d_f32,
    expand_1d_to_3d,
    squeeze_3d_to_1d,
    1,
    ConvOptions
);
conv_nd_via_3d!(
    conv1d_f64,
    conv3d_f64,
    expand_1d_to_3d,
    squeeze_3d_to_1d,
    1,
    ConvOptions
);
conv_nd_via_3d!(
    conv1d_f16,
    conv3d_f16,
    expand_1d_to_3d,
    squeeze_3d_to_1d,
    1,
    ConvOptions
);
bf16_via_f32!(conv1d_bf16, conv1d_f32, 1, ConvOptions);

fn expand_1d_to_3d(
    x: &FlexTensor,
    weight: &FlexTensor,
    options: &ConvOptions<1>,
) -> (FlexTensor, FlexTensor, ConvOptions<3>) {
    let x_shape = x.layout().shape();
    let x_3d = x.reshape(Shape::from(vec![x_shape[0], x_shape[1], 1, 1, x_shape[2]]));

    let w_shape = weight.layout().shape();
    let weight_3d = weight.reshape(Shape::from(vec![w_shape[0], w_shape[1], 1, 1, w_shape[2]]));

    let options_3d = ConvOptions::new(
        [1, 1, options.stride[0]],
        [0, 0, options.padding[0]],
        [1, 1, options.dilation[0]],
        options.groups,
    );

    (x_3d, weight_3d, options_3d)
}

// ============================================================================
// Conv2d - delegates to conv3d
// ============================================================================

conv_nd_via_3d!(
    conv2d_f32,
    conv3d_f32,
    expand_2d_to_3d,
    squeeze_3d_to_2d,
    2,
    ConvOptions
);
conv_nd_via_3d!(
    conv2d_f64,
    conv3d_f64,
    expand_2d_to_3d,
    squeeze_3d_to_2d,
    2,
    ConvOptions
);
conv_nd_via_3d!(
    conv2d_f16,
    conv3d_f16,
    expand_2d_to_3d,
    squeeze_3d_to_2d,
    2,
    ConvOptions
);
bf16_via_f32!(conv2d_bf16, conv2d_f32, 2, ConvOptions);

fn expand_2d_to_3d(
    x: &FlexTensor,
    weight: &FlexTensor,
    options: &ConvOptions<2>,
) -> (FlexTensor, FlexTensor, ConvOptions<3>) {
    let x_shape = x.layout().shape();
    let x_3d = x.reshape(Shape::from(vec![
        x_shape[0], x_shape[1], 1, x_shape[2], x_shape[3],
    ]));

    let w_shape = weight.layout().shape();
    let weight_3d = weight.reshape(Shape::from(vec![
        w_shape[0], w_shape[1], 1, w_shape[2], w_shape[3],
    ]));

    let options_3d = ConvOptions::new(
        [1, options.stride[0], options.stride[1]],
        [0, options.padding[0], options.padding[1]],
        [1, options.dilation[0], options.dilation[1]],
        options.groups,
    );

    (x_3d, weight_3d, options_3d)
}

// ============================================================================
// Conv3d - native implementations
// ============================================================================

conv3d_typed!(
    conv3d_f32,
    f32,
    DType::F32,
    0.0f32,
    gemm_f32,
    |a, b| a + b,
    conv3d_1x1_f32,
    conv3d_direct_f32
);
conv3d_typed!(
    conv3d_f64,
    f64,
    DType::F64,
    0.0f64,
    gemm_f64,
    |a, b| a + b,
    conv3d_1x1_f64,
    conv3d_direct_f64
);
conv3d_typed!(
    conv3d_f16,
    f16,
    DType::F16,
    f16::from_f32(0.0),
    gemm_f16,
    |a: f16, b: f16| f16::from_f32(a.to_f32() + b.to_f32()),
    conv3d_1x1_f16
);
bf16_via_f32!(conv3d_bf16, conv3d_f32, 3, ConvOptions);

/// Generic 3D convolution implementation using tiled im2col.
///
/// Tiled approach processes output in TILE_SIZE chunks:
/// - Reduces memory usage (smaller im2col buffer per tile)
/// - Enables tile-level parallelism
/// - Improves cache utilization
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn conv3d_impl<T: bytemuck::Pod + Clone + Copy + burn_backend::Element + Send + Sync>(
    x: FlexTensor,
    weight: FlexTensor,
    bias: Option<FlexTensor>,
    options: &ConvOptions<3>,
    dtype: DType,
    zero: T,
    gemm_fn: fn(&[T], &[T], usize, usize, usize) -> Vec<T>,
    add_fn: fn(T, T) -> T,
) -> FlexTensor {
    let x = x.to_contiguous();
    let weight = weight.to_contiguous();

    let x_shape = x.layout().shape();
    let w_shape = weight.layout().shape();

    let batch_size = x_shape[0];
    let channels_in = x_shape[1];
    let in_d = x_shape[2];
    let in_h = x_shape[3];
    let in_w = x_shape[4];

    let channels_out = w_shape[0];
    let channels_per_group = w_shape[1];
    let kernel_d = w_shape[2];
    let kernel_h = w_shape[3];
    let kernel_w = w_shape[4];

    let [stride_d, stride_h, stride_w] = options.stride;
    let [pad_d, pad_h, pad_w] = options.padding;
    let groups = options.groups;
    let out_channels_per_group = channels_out / groups;

    let out_d = calculate_conv_output_size(kernel_d, stride_d, pad_d, options.dilation[0], in_d);
    let out_h = calculate_conv_output_size(kernel_h, stride_h, pad_h, options.dilation[1], in_h);
    let out_w = calculate_conv_output_size(kernel_w, stride_w, pad_w, options.dilation[2], in_w);

    // Validate sizes won't overflow index calculations
    let _total = [batch_size, channels_out, out_d, out_h, out_w]
        .iter()
        .try_fold(1usize, |acc, &x| acc.checked_mul(x))
        .expect("conv: output tensor dimensions would overflow index calculations");
    let _col_total = [channels_per_group, kernel_d, kernel_h, kernel_w]
        .iter()
        .try_fold(1usize, |acc, &x| acc.checked_mul(x))
        .expect("conv: kernel dimensions would overflow index calculations");

    let x_data: &[T] = x.storage();
    let w_data: &[T] = weight.storage();

    let col_len = channels_per_group * kernel_d * kernel_h * kernel_w;
    let spatial_out = out_d * out_h * out_w;

    let [dilation_d, dilation_h, dilation_w] = options.dilation;

    // Tile size for processing output pixels. Larger = better GEMM utilization,
    // smaller = more parallelism and better cache usage. 512 is a good balance.
    const TILE_SIZE: usize = 512;
    let num_tiles = spatial_out.div_ceil(TILE_SIZE);

    // Flatten kernel [c_out, c_in, kd, kh, kw] -> [c_out, kd, kh, kw, c_in]
    // for GEMM. Expressed as a 2D transpose of (c_in, k_spatial) per c_out
    // so the compiler can unroll the k_spatial inner loop; the older 5-
    // nested formulation had a 10-term index expression LLVM wouldn't
    // autovectorize.
    let k_spatial = kernel_d * kernel_h * kernel_w;
    let mut w_flat = vec![zero; channels_out * col_len];
    for c_out in 0..channels_out {
        let src_base = c_out * channels_per_group * k_spatial;
        let dst_base = c_out * col_len;
        for c_in in 0..channels_per_group {
            let src_row = src_base + c_in * k_spatial;
            for k in 0..k_spatial {
                w_flat[dst_base + k * channels_per_group + c_in] = w_data[src_row + k];
            }
        }
    }

    // Convert input to NHWC layout for cache-friendly access in im2col.
    // Loop order (c innermost, spatial outer) is intentional: on aarch64
    // M3 Max, strided loads + contiguous stores beat the inverse by
    // 10-35% per layer. Load prefetchers outrun the store write buffer.
    let nhwc_stride = (
        in_d * in_h * in_w * channels_in,
        in_h * in_w * channels_in,
        in_w * channels_in,
        channels_in,
        1,
    );
    let mut x_nhwc = vec![zero; batch_size * in_d * in_h * in_w * channels_in];
    for b in 0..batch_size {
        for d in 0..in_d {
            for h in 0..in_h {
                for w in 0..in_w {
                    for c in 0..channels_in {
                        let src_idx = b * channels_in * in_d * in_h * in_w
                            + c * in_d * in_h * in_w
                            + d * in_h * in_w
                            + h * in_w
                            + w;
                        let dst_idx = b * nhwc_stride.0
                            + d * nhwc_stride.1
                            + h * nhwc_stride.2
                            + w * nhwc_stride.3
                            + c;
                        x_nhwc[dst_idx] = x_data[src_idx];
                    }
                }
            }
        }
    }

    // Use tiled parallel execution
    let output = {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;

            let mut dst = vec![zero; batch_size * channels_out * spatial_out];
            let dst_ptr = crate::ops::SendMutPtr::new(dst.as_mut_ptr());

            // Process batches and tiles in parallel (nested parallelism)
            (0..batch_size).into_par_iter().for_each(|b| {
                (0..num_tiles).into_par_iter().for_each(|tile_idx| {
                    let tile_start = tile_idx * TILE_SIZE;
                    let tile_end = (tile_start + TILE_SIZE).min(spatial_out);
                    let tile_size = tile_end - tile_start;

                    // Process each group separately
                    for g in 0..groups {
                        let in_c_start = g * channels_per_group;
                        let out_c_start = g * out_channels_per_group;

                        // Build im2col for this tile and group
                        let mut col_tile = vec![zero; col_len * tile_size];

                        im2col_3d_tile(
                            &mut col_tile,
                            &x_nhwc,
                            tile_start,
                            tile_end,
                            out_h,
                            out_w,
                            kernel_d,
                            kernel_h,
                            kernel_w,
                            stride_d,
                            stride_h,
                            stride_w,
                            dilation_d,
                            dilation_h,
                            dilation_w,
                            pad_d,
                            pad_h,
                            pad_w,
                            in_d,
                            in_h,
                            in_w,
                            channels_per_group,
                            col_len,
                            b,
                            in_c_start,
                            nhwc_stride,
                        );

                        // Get weight slice for this group
                        let w_start = out_c_start * col_len;
                        let w_end = w_start + out_channels_per_group * col_len;
                        let w_group = &w_flat[w_start..w_end];

                        // GEMM: w_group[out_c_per_group, col_len] @ col_tile[tile_size, col_len]^T
                        let result = gemm_fn(
                            w_group,
                            &col_tile,
                            out_channels_per_group,
                            col_len,
                            tile_size,
                        );

                        // Write results to output for this group's output channels
                        for (local_idx, global_idx) in (tile_start..tile_end).enumerate() {
                            for c_out in 0..out_channels_per_group {
                                let dst_idx = b * channels_out * spatial_out
                                    + (out_c_start + c_out) * spatial_out
                                    + global_idx;
                                let res_idx = c_out * tile_size + local_idx;
                                unsafe {
                                    debug_assert!(
                                        dst_idx < batch_size * channels_out * spatial_out
                                    );
                                    dst_ptr.write(dst_idx, result[res_idx]);
                                }
                            }
                        }
                    }
                });
            });
            dst
        }
        #[cfg(not(feature = "rayon"))]
        {
            // Sequential path with tiling
            let mut output = vec![zero; batch_size * channels_out * spatial_out];

            for b in 0..batch_size {
                for tile_idx in 0..num_tiles {
                    let tile_start = tile_idx * TILE_SIZE;
                    let tile_end = (tile_start + TILE_SIZE).min(spatial_out);
                    let tile_size = tile_end - tile_start;

                    // Process each group separately
                    for g in 0..groups {
                        let in_c_start = g * channels_per_group;
                        let out_c_start = g * out_channels_per_group;

                        let mut col_tile = vec![zero; col_len * tile_size];

                        im2col_3d_tile(
                            &mut col_tile,
                            &x_nhwc,
                            tile_start,
                            tile_end,
                            out_h,
                            out_w,
                            kernel_d,
                            kernel_h,
                            kernel_w,
                            stride_d,
                            stride_h,
                            stride_w,
                            dilation_d,
                            dilation_h,
                            dilation_w,
                            pad_d,
                            pad_h,
                            pad_w,
                            in_d,
                            in_h,
                            in_w,
                            channels_per_group,
                            col_len,
                            b,
                            in_c_start,
                            nhwc_stride,
                        );

                        // Get weight slice for this group
                        let w_start = out_c_start * col_len;
                        let w_end = w_start + out_channels_per_group * col_len;
                        let w_group = &w_flat[w_start..w_end];

                        // GEMM for this group
                        let result = gemm_fn(
                            w_group,
                            &col_tile,
                            out_channels_per_group,
                            col_len,
                            tile_size,
                        );

                        // Write results to output for this group's output channels
                        for (local_idx, global_idx) in (tile_start..tile_end).enumerate() {
                            for c_out in 0..out_channels_per_group {
                                let dst_idx = b * channels_out * spatial_out
                                    + (out_c_start + c_out) * spatial_out
                                    + global_idx;
                                let res_idx = c_out * tile_size + local_idx;
                                output[dst_idx] = result[res_idx];
                            }
                        }
                    }
                }
            }
            output
        }
    };

    let mut output = output;
    if let Some(bias) = bias {
        let bias = bias.to_contiguous();
        let bias_data: &[T] = bias.storage();
        add_bias(
            &mut output,
            bias_data,
            batch_size,
            channels_out,
            spatial_out,
            add_fn,
        );
    }

    let out_shape = Shape::from(vec![batch_size, channels_out, out_d, out_h, out_w]);
    FlexTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(out_shape),
        dtype,
    )
}

/// Build im2col tile for a range of output positions.
///
/// Fills `col_tile` with shape [tile_size, col_len] where each row is a flattened
/// patch from the NHWC input for one output position.
#[allow(clippy::too_many_arguments)]
fn im2col_3d_tile<T: bytemuck::Pod + Copy>(
    col_tile: &mut [T],
    x_nhwc: &[T],
    tile_start: usize,
    tile_end: usize,
    out_h: usize,
    out_w: usize,
    kernel_d: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_d: usize,
    stride_h: usize,
    stride_w: usize,
    dilation_d: usize,
    dilation_h: usize,
    dilation_w: usize,
    pad_d: usize,
    pad_h: usize,
    pad_w: usize,
    in_d: usize,
    in_h: usize,
    in_w: usize,
    channels_per_group: usize,
    col_len: usize,
    b: usize,
    in_c_start: usize,
    nhwc_stride: (usize, usize, usize, usize, usize),
) {
    for (local_idx, global_idx) in (tile_start..tile_end).enumerate() {
        let out_d_idx = global_idx / (out_h * out_w);
        let rem = global_idx % (out_h * out_w);
        let out_h_idx = rem / out_w;
        let out_w_idx = rem % out_w;

        let mut col_offset = 0;
        for kd in 0..kernel_d {
            let id = (out_d_idx * stride_d + kd * dilation_d) as isize - pad_d as isize;
            for kh in 0..kernel_h {
                let ih = (out_h_idx * stride_h + kh * dilation_h) as isize - pad_h as isize;
                for kw in 0..kernel_w {
                    let iw = (out_w_idx * stride_w + kw * dilation_w) as isize - pad_w as isize;

                    if id >= 0
                        && id < in_d as isize
                        && ih >= 0
                        && ih < in_h as isize
                        && iw >= 0
                        && iw < in_w as isize
                    {
                        let id = id as usize;
                        let ih = ih as usize;
                        let iw = iw as usize;
                        let inp_base = b * nhwc_stride.0
                            + id * nhwc_stride.1
                            + ih * nhwc_stride.2
                            + iw * nhwc_stride.3
                            + in_c_start;
                        for c in 0..channels_per_group {
                            col_tile[local_idx * col_len + col_offset] = x_nhwc[inp_base + c];
                            col_offset += 1;
                        }
                    } else {
                        // Padding: skip channels_per_group positions (already zero)
                        col_offset += channels_per_group;
                    }
                }
            }
        }
    }
}

/// Check if this is a 1x1 convolution that can use the fast path.
fn is_1x1_conv(
    kernel_d: usize,
    kernel_h: usize,
    kernel_w: usize,
    options: &ConvOptions<3>,
) -> bool {
    kernel_d == 1
        && kernel_h == 1
        && kernel_w == 1
        && options.stride == [1, 1, 1]
        && options.padding == [0, 0, 0]
}

/// Optimized 1x1 convolution: skip im2col, call gemm directly on NCHW data.
///
/// For 1x1 conv with stride=1 and padding=0, the input for each (batch, group) is
/// already [channels_per_group, spatial] row-major in NCHW layout. We pass it directly
/// to gemm as the RHS with appropriate strides, avoiding the transpose allocation
/// and the intermediate result buffer.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn conv3d_1x1_impl<T: bytemuck::Pod + Clone + Copy + burn_backend::Element + Send + Sync>(
    x: FlexTensor,
    weight: FlexTensor,
    bias: Option<FlexTensor>,
    options: &ConvOptions<3>,
    dtype: DType,
    zero: T,
    one: T,
    add_fn: fn(T, T) -> T,
) -> FlexTensor {
    let x = x.to_contiguous();
    let weight = weight.to_contiguous();

    let x_shape = x.layout().shape();
    let w_shape = weight.layout().shape();

    let batch_size = x_shape[0];
    let channels_in = x_shape[1];
    let spatial = x_shape[2] * x_shape[3] * x_shape[4];

    let channels_out = w_shape[0];
    let channels_per_group = w_shape[1];
    let groups = options.groups;
    let out_channels_per_group = channels_out / groups;

    let m = out_channels_per_group;
    let k = channels_per_group;
    let n = spatial;

    // Validate output size won't overflow
    let total_output = [batch_size, channels_out, spatial]
        .iter()
        .try_fold(1usize, |acc, &x| acc.checked_mul(x))
        .expect("conv 1x1: output tensor dimensions would overflow index calculations");

    let x_data: &[T] = x.storage();
    let w_data: &[T] = weight.storage();

    // Gemm: C[M,N] = W[M,K] * X[K,N]
    // W is [out_channels_per_group, channels_per_group] row-major
    // X is [channels_per_group, spatial] row-major (directly from NCHW layout)
    // C is [out_channels_per_group, spatial] row-major
    #[cfg(feature = "rayon")]
    let parallelism = if m.saturating_mul(n).saturating_mul(k) >= 192 * 192 * 192 {
        gemm::Parallelism::Rayon(0)
    } else {
        gemm::Parallelism::None
    };
    #[cfg(not(feature = "rayon"))]
    let parallelism = gemm::Parallelism::None;

    let mut output = vec![zero; total_output];

    {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            let dst_ptr = crate::ops::SendMutPtr::new(output.as_mut_ptr());

            (0..batch_size).into_par_iter().for_each(|b| {
                for g in 0..groups {
                    let x_offset = b * channels_in * spatial + g * k * spatial;
                    let w_offset = g * out_channels_per_group * k;
                    let out_offset =
                        b * channels_out * spatial + g * out_channels_per_group * spatial;

                    unsafe {
                        gemm::gemm(
                            m,
                            n,
                            k,
                            dst_ptr.ptr_add(out_offset),
                            1,          // dst_cs
                            n as isize, // dst_rs
                            false,      // read_dst
                            w_data.as_ptr().add(w_offset),
                            1,          // lhs_cs: W row-major
                            k as isize, // lhs_rs
                            x_data.as_ptr().add(x_offset),
                            1,          // rhs_cs: X row-major
                            n as isize, // rhs_rs
                            zero,
                            one,
                            false,
                            false,
                            false,
                            parallelism,
                        );
                    }
                }
            });
        }
        #[cfg(not(feature = "rayon"))]
        {
            for b in 0..batch_size {
                for g in 0..groups {
                    let x_offset = b * channels_in * spatial + g * k * spatial;
                    let w_offset = g * out_channels_per_group * k;
                    let out_offset =
                        b * channels_out * spatial + g * out_channels_per_group * spatial;

                    unsafe {
                        gemm::gemm(
                            m,
                            n,
                            k,
                            output.as_mut_ptr().add(out_offset),
                            1,
                            n as isize,
                            false,
                            w_data.as_ptr().add(w_offset),
                            1,
                            k as isize,
                            x_data.as_ptr().add(x_offset),
                            1,
                            n as isize,
                            zero,
                            one,
                            false,
                            false,
                            false,
                            parallelism,
                        );
                    }
                }
            }
        }
    }

    if let Some(bias) = bias {
        let bias = bias.to_contiguous();
        let bias_data: &[T] = bias.storage();
        add_bias(
            &mut output,
            bias_data,
            batch_size,
            channels_out,
            spatial,
            add_fn,
        );
    }

    let out_shape = Shape::from(vec![
        batch_size,
        channels_out,
        x_shape[2],
        x_shape[3],
        x_shape[4],
    ]);
    FlexTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(out_shape),
        dtype,
    )
}

conv3d_1x1_typed!(conv3d_1x1_f32, f32, DType::F32, 0.0f32, 1.0f32, |a, b| a
    + b);
conv3d_1x1_typed!(conv3d_1x1_f64, f64, DType::F64, 0.0f64, 1.0f64, |a, b| a
    + b);
conv3d_1x1_typed!(
    conv3d_1x1_f16,
    f16,
    DType::F16,
    f16::from_f32(0.0),
    f16::from_f32(1.0),
    |a: f16, b: f16| f16::from_f32(a.to_f32() + b.to_f32())
);

// ============================================================================
// Direct conv path for small spatial sizes (no im2col)
// ============================================================================

/// Decide whether to use the direct conv path instead of tiled im2col + GEMM.
///
/// The direct path decomposes the conv into kw gemm calls with strided pointers
/// into NCHW data, eliminating both the NHWC conversion and im2col buffer.
/// This wins when spatial_out is small enough that im2col allocation and fill
/// overhead is significant relative to compute.
fn should_use_direct_conv(x_shape: &[usize], w_shape: &[usize], options: &ConvOptions<3>) -> bool {
    // Only for groups=1, no padding, dilation=1 (the wav2vec2 case).
    if options.groups != 1 || options.padding != [0, 0, 0] || options.dilation != [1, 1, 1] {
        return false;
    }

    let kernel_d = w_shape[2];
    let kernel_h = w_shape[3];
    let kernel_w = w_shape[4];

    // Only for "1D-like" convolutions expanded to 3D
    if kernel_d != 1 || kernel_h != 1 {
        return false;
    }
    if x_shape[2] != 1 || x_shape[3] != 1 {
        return false;
    }

    let channels_in = x_shape[1];
    let in_w = x_shape[4];
    let out_w = calculate_conv_output_size(kernel_w, options.stride[2], 0, 1, in_w);

    // The direct path eliminates im2col buffer allocation (kw * c_in * tile_size floats)
    // and the copy into it. This matters when spatial_out is small relative to c_in * kw.
    channels_in >= 32 && kernel_w <= 8 && out_w <= 800
}

macro_rules! conv3d_direct_typed {
    ($fn_name:ident, $T:ty, $dtype:expr, $zero:expr, $one:expr, $add_fn:expr) => {
        fn $fn_name(
            x: FlexTensor,
            weight: FlexTensor,
            bias: Option<FlexTensor>,
            options: &ConvOptions<3>,
        ) -> FlexTensor {
            conv3d_direct_impl::<$T>(x, weight, bias, options, $dtype, $zero, $one, $add_fn)
        }
    };
}

conv3d_direct_typed!(
    conv3d_direct_f32,
    f32,
    DType::F32,
    0.0f32,
    1.0f32,
    |a, b| a + b
);
conv3d_direct_typed!(
    conv3d_direct_f64,
    f64,
    DType::F64,
    0.0f64,
    1.0f64,
    |a, b| a + b
);

/// Direct conv3d: decompose into kw gemm calls on NCHW data directly.
///
/// The conv operation is: out[co, o] = sum_k sum_c w[co, c, k] * x[c, o*stride+k]
///
/// This decomposes into kw matrix multiplies, one per kernel position:
///   out += W_k @ X_k  where W_k = w[:, :, k] and X_k = x[:, o*stride+k]
///
/// Each gemm reads directly from NCHW input with strided pointers, eliminating
/// both the NHWC conversion and im2col buffer entirely.
///
/// Constraints: groups=1, padding=0, dilation=1, 1D-like (d=1, h=1).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn conv3d_direct_impl<T: bytemuck::Pod + Clone + Copy + burn_backend::Element + Send + Sync>(
    x: FlexTensor,
    weight: FlexTensor,
    bias: Option<FlexTensor>,
    options: &ConvOptions<3>,
    dtype: DType,
    zero: T,
    one: T,
    add_fn: fn(T, T) -> T,
) -> FlexTensor {
    let x = x.to_contiguous();
    let weight = weight.to_contiguous();

    let x_shape = x.layout().shape();
    let w_shape = weight.layout().shape();

    let batch_size = x_shape[0];
    let channels_in = x_shape[1];
    let in_w = x_shape[4];

    let channels_out = w_shape[0];
    let kernel_w = w_shape[4];
    let stride_w = options.stride[2];

    let out_w = calculate_conv_output_size(kernel_w, stride_w, 0, 1, in_w);

    let x_data: &[T] = x.storage();
    let w_data: &[T] = weight.storage();

    // Weight layout: [c_out, c_in, kw] contiguous (kd=kh=1).
    // For kernel position k: W_k[co, c] = w_data[co * c_in * kw + c * kw + k]
    let lhs_rs = (channels_in * kernel_w) as isize;
    let lhs_cs = kernel_w as isize;

    // For kernel position k: X_k[c_in, o] = x_data[c_in * in_w + o * stride + k]
    let rhs_rs = in_w as isize;
    let rhs_cs = stride_w as isize;

    let m = channels_out;
    let gemm_k = channels_in;
    let n = out_w;

    #[cfg(feature = "rayon")]
    let parallelism = if m.saturating_mul(n).saturating_mul(gemm_k) >= 192 * 192 * 192 {
        gemm::Parallelism::Rayon(0)
    } else {
        gemm::Parallelism::None
    };
    #[cfg(not(feature = "rayon"))]
    let parallelism = gemm::Parallelism::None;

    let total_output = batch_size
        .checked_mul(channels_out)
        .and_then(|x| x.checked_mul(out_w))
        .expect("conv direct: output dimensions overflow");
    let mut output = vec![zero; total_output];

    let batch_x_len = channels_in * in_w;

    {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            let dst_ptr = crate::ops::SendMutPtr::new(output.as_mut_ptr());

            (0..batch_size).into_par_iter().for_each(|b| {
                let out_offset = b * channels_out * out_w;

                for k in 0..kernel_w {
                    unsafe {
                        let x_base = x_data.as_ptr().add(b * batch_x_len);
                        gemm::gemm(
                            m,
                            n,
                            gemm_k,
                            dst_ptr.ptr_add(out_offset),
                            1,
                            n as isize,
                            k > 0,
                            w_data.as_ptr().add(k),
                            lhs_cs,
                            lhs_rs,
                            x_base.add(k),
                            rhs_cs,
                            rhs_rs,
                            one,
                            one,
                            false,
                            false,
                            false,
                            parallelism,
                        );
                    }
                }
            });
        }
        #[cfg(not(feature = "rayon"))]
        {
            for b in 0..batch_size {
                let out_offset = b * channels_out * out_w;

                for k in 0..kernel_w {
                    unsafe {
                        let x_base = x_data.as_ptr().add(b * batch_x_len);
                        gemm::gemm(
                            m,
                            n,
                            gemm_k,
                            output.as_mut_ptr().add(out_offset),
                            1,
                            n as isize,
                            k > 0,
                            w_data.as_ptr().add(k),
                            lhs_cs,
                            lhs_rs,
                            x_base.add(k),
                            rhs_cs,
                            rhs_rs,
                            one,
                            one,
                            false,
                            false,
                            false,
                            parallelism,
                        );
                    }
                }
            }
        }
    }

    if let Some(bias) = bias {
        let bias = bias.to_contiguous();
        let bias_data: &[T] = bias.storage();
        add_bias(
            &mut output,
            bias_data,
            batch_size,
            channels_out,
            out_w,
            add_fn,
        );
    }

    let out_shape = Shape::from(vec![batch_size, channels_out, 1, 1, out_w]);
    FlexTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(out_shape),
        dtype,
    )
}

// ============================================================================
// gemm implementations
// ============================================================================
//
// One thin wrapper per element type around `gemm::gemm` with the strides and
// parallelism heuristic fixed to what conv3d_impl needs. The wrappers share
// every line of logic aside from the numeric type, zero, and one literals,
// so they're generated via a macro.

macro_rules! gemm_typed {
    ($fn_name:ident, $T:ty, $zero:expr, $one:expr) => {
        fn $fn_name(a: &[$T], b: &[$T], m: usize, k: usize, n: usize) -> Vec<$T> {
            let mut c = vec![$zero; m * n];
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
                    1,
                    n as isize,
                    false,
                    a.as_ptr(),
                    1,
                    k as isize,
                    b.as_ptr(),
                    k as isize,
                    1,
                    $zero,
                    $one,
                    false,
                    false,
                    false,
                    parallelism,
                );
            }
            c
        }
    };
}

gemm_typed!(gemm_f32, f32, 0.0f32, 1.0f32);
gemm_typed!(gemm_f64, f64, 0.0f64, 1.0f64);
gemm_typed!(gemm_f16, f16, f16::from_f32(0.0), f16::from_f32(1.0));

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::TensorData;
    use burn_std::bf16;

    #[test]
    fn test_conv2d_1x1() {
        // 1x1 convolution uses the optimized fast path (no im2col)
        // Input: [1, 4, 3, 3] (batch=1, channels=4, 3x3 spatial)
        // Weight: [8, 4, 1, 1] (8 output channels, 4 input channels, 1x1 kernel)
        // Output: [1, 8, 3, 3]
        let x_data: Vec<f32> = (0..36).map(|x| x as f32).collect();
        let x = FlexTensor::from_data(TensorData::new(x_data, vec![1, 4, 3, 3]));

        // Weight: each output channel sums specific input channels
        // Simple weight: first output channel = sum of all input channels
        let mut w_data = vec![0.0f32; 32]; // 8 * 4 = 32
        for i in 0..4 {
            w_data[i] = 1.0; // First output channel: sum all inputs
        }
        w_data[4] = 1.0; // Second output channel: just first input channel
        let weight = FlexTensor::from_data(TensorData::new(w_data, vec![8, 4, 1, 1]));

        let options = ConvOptions::new([1, 1], [0, 0], [1, 1], 1);
        let result = conv2d_f32(x, weight, None, &options);

        assert_eq!(result.layout().shape().to_vec(), vec![1, 8, 3, 3]);
        let out: Vec<f32> = result.into_data().to_vec().unwrap();

        // First output channel should be sum across all 4 input channels at each position
        // Position (0,0): channels 0-3 at (0,0) = 0 + 9 + 18 + 27 = 54
        assert!((out[0] - 54.0).abs() < 1e-5, "got {}", out[0]);

        // Second output channel should be just the first input channel
        // Position (0,0): channel 0 at (0,0) = 0
        let second_ch_start = 9; // 3*3 = 9 elements per channel
        assert!(
            (out[second_ch_start] - 0.0).abs() < 1e-5,
            "got {}",
            out[second_ch_start]
        );
    }

    #[test]
    fn test_conv2d_1x1_with_bias() {
        // 1x1 conv with bias
        let x = FlexTensor::from_data(TensorData::new(vec![1.0f32; 16], vec![1, 4, 2, 2]));
        let w_data: Vec<f32> = (0..8).map(|_| 0.5f32).collect(); // 2 output channels, 4 input
        let weight = FlexTensor::from_data(TensorData::new(w_data, vec![2, 4, 1, 1]));
        let bias = FlexTensor::from_data(TensorData::new(vec![10.0f32, 20.0f32], vec![2]));

        let options = ConvOptions::new([1, 1], [0, 0], [1, 1], 1);
        let result = conv2d_f32(x, weight, Some(bias), &options);

        let out: Vec<f32> = result.into_data().to_vec().unwrap();
        // Each output = 4 * 1.0 * 0.5 + bias = 2.0 + bias
        assert!((out[0] - 12.0).abs() < 1e-5); // First channel: 2.0 + 10.0
        assert!((out[4] - 22.0).abs() < 1e-5); // Second channel: 2.0 + 20.0
    }

    #[test]
    fn test_conv2d_1x1_groups() {
        // 1x1 conv with groups=2: 4 input channels split into 2 groups of 2
        // Weight: [4, 2, 1, 1] (4 output channels, 2 per group)
        let x_data: Vec<f32> = (0..16).map(|x| x as f32).collect();
        let x = FlexTensor::from_data(TensorData::new(x_data, vec![1, 4, 2, 2]));

        // Group 0: out channels 0-1 from in channels 0-1
        // Group 1: out channels 2-3 from in channels 2-3
        let mut w_data = vec![0.0f32; 8]; // 4 * 2 = 8
        w_data[0] = 1.0; // out_ch 0 = 1.0 * in_ch 0
        w_data[1] = 0.0;
        w_data[2] = 0.0;
        w_data[3] = 1.0; // out_ch 1 = 1.0 * in_ch 1
        w_data[4] = 1.0; // out_ch 2 = 1.0 * in_ch 2
        w_data[5] = 1.0; // out_ch 2 += 1.0 * in_ch 3
        w_data[6] = 0.0;
        w_data[7] = 0.0; // out_ch 3 = 0
        let weight = FlexTensor::from_data(TensorData::new(w_data, vec![4, 2, 1, 1]));

        let options = ConvOptions::new([1, 1], [0, 0], [1, 1], 2);
        let result = conv2d_f32(x, weight, None, &options);

        assert_eq!(result.layout().shape().to_vec(), vec![1, 4, 2, 2]);
        let out: Vec<f32> = result.into_data().to_vec().unwrap();

        // out_ch 0 = in_ch 0: [0, 1, 2, 3]
        assert_eq!(&out[0..4], &[0.0, 1.0, 2.0, 3.0]);
        // out_ch 1 = in_ch 1: [4, 5, 6, 7]
        assert_eq!(&out[4..8], &[4.0, 5.0, 6.0, 7.0]);
        // out_ch 2 = in_ch 2 + in_ch 3: [8+12, 9+13, 10+14, 11+15]
        assert_eq!(&out[8..12], &[20.0, 22.0, 24.0, 26.0]);
        // out_ch 3 = 0
        assert_eq!(&out[12..16], &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_conv2d_1x1_groups_with_bias() {
        // 1x1 grouped conv with bias
        let x = FlexTensor::from_data(TensorData::new(vec![1.0f32; 16], vec![1, 4, 2, 2]));
        let w_data = vec![0.5f32; 8]; // 4 out channels, 2 per group
        let weight = FlexTensor::from_data(TensorData::new(w_data, vec![4, 2, 1, 1]));
        let bias = FlexTensor::from_data(TensorData::new(vec![10.0f32, 20.0, 30.0, 40.0], vec![4]));

        let options = ConvOptions::new([1, 1], [0, 0], [1, 1], 2);
        let result = conv2d_f32(x, weight, Some(bias), &options);

        let out: Vec<f32> = result.into_data().to_vec().unwrap();
        // Each output = 2 * 1.0 * 0.5 + bias = 1.0 + bias
        assert!((out[0] - 11.0).abs() < 1e-5); // ch 0: 1.0 + 10.0
        assert!((out[4] - 21.0).abs() < 1e-5); // ch 1: 1.0 + 20.0
        assert!((out[8] - 31.0).abs() < 1e-5); // ch 2: 1.0 + 30.0
        assert!((out[12] - 41.0).abs() < 1e-5); // ch 3: 1.0 + 40.0
    }

    #[test]
    fn test_conv1d_simple() {
        let x_data: Vec<f32> = (1..=5).map(|x| x as f32).collect();
        let x = FlexTensor::from_data(TensorData::new(x_data, vec![1, 1, 5]));
        let w_data = vec![1.0f32, 1.0, 1.0];
        let weight = FlexTensor::from_data(TensorData::new(w_data, vec![1, 1, 3]));
        let options = ConvOptions::new([1], [0], [1], 1);
        let result = conv1d_f32(x, weight, None, &options);
        assert_eq!(result.layout().shape().to_vec(), vec![1, 1, 3]);
        let out: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(out, vec![6.0, 9.0, 12.0]);
    }

    #[test]
    fn test_conv1d_direct_path() {
        // c_in=64 >= 32, kernel=3, stride=2, out_w=49 <= 800 -> hits direct path
        let c_in = 64;
        let c_out = 32;
        let in_w = 100;
        let kw = 3;
        let stride = 2;
        let out_w = (in_w - kw) / stride + 1; // 49

        // Deterministic input and weights
        let x_data: Vec<f32> = (0..c_in * in_w)
            .map(|i| ((i % 100) as f32 / 100.0) - 0.5)
            .collect();
        let w_data: Vec<f32> = (0..c_out * c_in * kw)
            .map(|i| ((i % 50) as f32 / 50.0) - 0.5)
            .collect();

        let x = FlexTensor::from_data(TensorData::new(x_data.clone(), vec![1, c_in, in_w]));
        let weight = FlexTensor::from_data(TensorData::new(w_data.clone(), vec![c_out, c_in, kw]));
        let options = ConvOptions::new([stride], [0], [1], 1);
        let result = conv1d_f32(x, weight, None, &options);
        let out: Vec<f32> = result.into_data().to_vec().unwrap();

        assert_eq!(out.len(), c_out * out_w);

        // Verify against naive reference implementation
        for co in 0..c_out {
            for o in 0..out_w {
                let mut expected = 0.0f32;
                for ci in 0..c_in {
                    for k in 0..kw {
                        expected += w_data[co * c_in * kw + ci * kw + k]
                            * x_data[ci * in_w + o * stride + k];
                    }
                }
                let actual = out[co * out_w + o];
                assert!(
                    (actual - expected).abs() < 1e-3,
                    "mismatch at co={co}, o={o}: expected {expected}, got {actual}"
                );
            }
        }
    }

    #[test]
    fn test_conv1d_direct_path_kw2() {
        // Test with kw=2 (L5/L6 shapes use kw=2)
        let c_in = 64;
        let c_out = 32;
        let in_w = 50;
        let kw = 2;
        let stride = 2;
        let out_w = (in_w - kw) / stride + 1; // 24

        let x_data: Vec<f32> = (0..c_in * in_w)
            .map(|i| ((i % 100) as f32 / 100.0) - 0.5)
            .collect();
        let w_data: Vec<f32> = (0..c_out * c_in * kw)
            .map(|i| ((i % 50) as f32 / 50.0) - 0.5)
            .collect();

        let x = FlexTensor::from_data(TensorData::new(x_data.clone(), vec![1, c_in, in_w]));
        let weight = FlexTensor::from_data(TensorData::new(w_data.clone(), vec![c_out, c_in, kw]));
        let options = ConvOptions::new([stride], [0], [1], 1);
        let result = conv1d_f32(x, weight, None, &options);
        let out: Vec<f32> = result.into_data().to_vec().unwrap();

        for co in 0..c_out {
            for o in 0..out_w {
                let mut expected = 0.0f32;
                for ci in 0..c_in {
                    for k in 0..kw {
                        expected += w_data[co * c_in * kw + ci * kw + k]
                            * x_data[ci * in_w + o * stride + k];
                    }
                }
                let actual = out[co * out_w + o];
                assert!(
                    (actual - expected).abs() < 1e-3,
                    "mismatch at co={co}, o={o}: expected {expected}, got {actual}"
                );
            }
        }
    }

    #[test]
    fn test_conv1d_direct_path_f64() {
        let c_in = 64;
        let c_out = 16;
        let in_w = 50;
        let kw = 3;
        let stride = 2;
        let out_w = (in_w - kw) / stride + 1;

        let x_data: Vec<f64> = (0..c_in * in_w)
            .map(|i| ((i % 100) as f64 / 100.0) - 0.5)
            .collect();
        let w_data: Vec<f64> = (0..c_out * c_in * kw)
            .map(|i| ((i % 50) as f64 / 50.0) - 0.5)
            .collect();

        let x = FlexTensor::from_data(TensorData::new(x_data.clone(), vec![1, c_in, in_w]));
        let weight = FlexTensor::from_data(TensorData::new(w_data.clone(), vec![c_out, c_in, kw]));
        let options = ConvOptions::new([stride], [0], [1], 1);
        let result = conv1d_f64(x, weight, None, &options);
        let out: Vec<f64> = result.into_data().to_vec().unwrap();

        for co in 0..c_out {
            for o in 0..out_w {
                let mut expected = 0.0f64;
                for ci in 0..c_in {
                    for k in 0..kw {
                        expected += w_data[co * c_in * kw + ci * kw + k]
                            * x_data[ci * in_w + o * stride + k];
                    }
                }
                let actual = out[co * out_w + o];
                assert!(
                    (actual - expected).abs() < 1e-10,
                    "f64 mismatch at co={co}, o={o}: expected {expected}, got {actual}"
                );
            }
        }
    }

    #[test]
    fn test_conv1d_direct_path_with_bias() {
        let c_in = 64;
        let c_out = 32;
        let in_w = 50;
        let kw = 2;
        let stride = 2;
        let out_w = (in_w - kw) / stride + 1;

        let x_data: Vec<f32> = (0..c_in * in_w)
            .map(|i| ((i % 100) as f32 / 100.0) - 0.5)
            .collect();
        let w_data: Vec<f32> = (0..c_out * c_in * kw)
            .map(|i| ((i % 50) as f32 / 50.0) - 0.5)
            .collect();
        let bias_data: Vec<f32> = (0..c_out).map(|i| i as f32 * 0.1).collect();

        let x = FlexTensor::from_data(TensorData::new(x_data.clone(), vec![1, c_in, in_w]));
        let weight = FlexTensor::from_data(TensorData::new(w_data.clone(), vec![c_out, c_in, kw]));
        let bias = FlexTensor::from_data(TensorData::new(bias_data.clone(), vec![c_out]));
        let options = ConvOptions::new([stride], [0], [1], 1);
        let result = conv1d_f32(x, weight, Some(bias), &options);
        let out: Vec<f32> = result.into_data().to_vec().unwrap();

        for co in 0..c_out {
            for o in 0..out_w {
                let mut expected = bias_data[co];
                for ci in 0..c_in {
                    for k in 0..kw {
                        expected += w_data[co * c_in * kw + ci * kw + k]
                            * x_data[ci * in_w + o * stride + k];
                    }
                }
                let actual = out[co * out_w + o];
                assert!(
                    (actual - expected).abs() < 1e-3,
                    "bias mismatch at co={co}, o={o}: expected {expected}, got {actual}"
                );
            }
        }
    }

    #[test]
    fn test_conv2d_simple() {
        let x_data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let x = FlexTensor::from_data(TensorData::new(x_data, vec![1, 1, 4, 4]));
        let w_data = vec![1.0f32; 4];
        let weight = FlexTensor::from_data(TensorData::new(w_data, vec![1, 1, 2, 2]));
        let options = ConvOptions::new([1, 1], [0, 0], [1, 1], 1);
        let result = conv2d_f32(x, weight, None, &options);
        assert_eq!(result.layout().shape().to_vec(), vec![1, 1, 3, 3]);
        let out: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(
            out,
            vec![14.0, 18.0, 22.0, 30.0, 34.0, 38.0, 46.0, 50.0, 54.0]
        );
    }

    #[test]
    fn test_conv2d_with_padding() {
        let x_data: Vec<f32> = (1..=9).map(|x| x as f32).collect();
        let x = FlexTensor::from_data(TensorData::new(x_data, vec![1, 1, 3, 3]));
        let w_data = vec![1.0f32; 9];
        let weight = FlexTensor::from_data(TensorData::new(w_data, vec![1, 1, 3, 3]));
        let options = ConvOptions::new([1, 1], [1, 1], [1, 1], 1);
        let result = conv2d_f32(x, weight, None, &options);
        let out: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(out[4], 45.0); // center element sums all
    }

    #[test]
    fn test_conv2d_with_bias() {
        let x = FlexTensor::from_data(TensorData::new(vec![1.0f32; 16], vec![1, 1, 4, 4]));
        let weight = FlexTensor::from_data(TensorData::new(vec![1.0f32; 4], vec![1, 1, 2, 2]));
        let bias = FlexTensor::from_data(TensorData::new(vec![10.0f32], vec![1]));
        let options = ConvOptions::new([1, 1], [0, 0], [1, 1], 1);
        let result = conv2d_f32(x, weight, Some(bias), &options);
        let out: Vec<f32> = result.into_data().to_vec().unwrap();
        assert!(out.iter().all(|&v| (v - 14.0).abs() < 1e-5));
    }

    #[test]
    fn test_conv2d_groups() {
        let x = FlexTensor::from_data(TensorData::new(vec![1.0f32; 36], vec![1, 4, 3, 3]));
        let weight = FlexTensor::from_data(TensorData::new(vec![1.0f32; 32], vec![4, 2, 2, 2]));
        let options = ConvOptions::new([1, 1], [0, 0], [1, 1], 2);
        let result = conv2d_f32(x, weight, None, &options);
        assert_eq!(result.layout().shape().to_vec(), vec![1, 4, 2, 2]);
    }

    #[test]
    fn test_conv3d_simple() {
        let x = FlexTensor::from_data(TensorData::new(vec![1.0f32; 18], vec![1, 1, 2, 3, 3]));
        let weight = FlexTensor::from_data(TensorData::new(vec![1.0f32; 8], vec![1, 1, 2, 2, 2]));
        let options = ConvOptions::new([1, 1, 1], [0, 0, 0], [1, 1, 1], 1);
        let result = conv3d_f32(x, weight, None, &options);
        assert_eq!(result.layout().shape().to_vec(), vec![1, 1, 1, 2, 2]);
        let out: Vec<f32> = result.into_data().to_vec().unwrap();
        assert!(out.iter().all(|&v| (v - 8.0).abs() < 1e-5));
    }

    #[test]
    fn test_conv2d_f64() {
        let x_data: Vec<f64> = (1..=16).map(|x| x as f64).collect();
        let x = FlexTensor::from_data(TensorData::new(x_data, vec![1, 1, 4, 4]));
        let w_data = vec![1.0f64; 4];
        let weight = FlexTensor::from_data(TensorData::new(w_data, vec![1, 1, 2, 2]));
        let options = ConvOptions::new([1, 1], [0, 0], [1, 1], 1);
        let result = conv2d_f64(x, weight, None, &options);
        let out: Vec<f64> = result.into_data().to_vec().unwrap();
        assert_eq!(
            out,
            vec![14.0, 18.0, 22.0, 30.0, 34.0, 38.0, 46.0, 50.0, 54.0]
        );
    }

    #[test]
    fn test_conv2d_f16() {
        let x_data: Vec<f16> = (1..=16).map(|x| f16::from_f32(x as f32)).collect();
        let x = FlexTensor::from_data(TensorData::new(x_data, vec![1, 1, 4, 4]));
        let w_data: Vec<f16> = vec![f16::from_f32(1.0); 4];
        let weight = FlexTensor::from_data(TensorData::new(w_data, vec![1, 1, 2, 2]));
        let options = ConvOptions::new([1, 1], [0, 0], [1, 1], 1);
        let result = conv2d_f16(x, weight, None, &options);
        let out: Vec<f16> = result.into_data().to_vec().unwrap();
        let expected = vec![14.0, 18.0, 22.0, 30.0, 34.0, 38.0, 46.0, 50.0, 54.0];
        for (a, e) in out.iter().zip(expected.iter()) {
            assert!((a.to_f32() - e).abs() < 0.5);
        }
    }

    #[test]
    fn test_conv2d_bf16() {
        let x_data: Vec<bf16> = (1..=16).map(|x| bf16::from_f32(x as f32)).collect();
        let x = FlexTensor::from_data(TensorData::new(x_data, vec![1, 1, 4, 4]));
        let w_data: Vec<bf16> = vec![bf16::from_f32(1.0); 4];
        let weight = FlexTensor::from_data(TensorData::new(w_data, vec![1, 1, 2, 2]));
        let options = ConvOptions::new([1, 1], [0, 0], [1, 1], 1);
        let result = conv2d_bf16(x, weight, None, &options);
        let out: Vec<bf16> = result.into_data().to_vec().unwrap();
        let expected = vec![14.0, 18.0, 22.0, 30.0, 34.0, 38.0, 46.0, 50.0, 54.0];
        for (a, e) in out.iter().zip(expected.iter()) {
            assert!((a.to_f32() - e).abs() < 0.5);
        }
    }
}
