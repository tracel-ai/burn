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
//! - Depthwise fast path: For canonical depthwise (groups == c_in == c_out,
//!   channels_per_group == 1), skip NHWC conversion, im2col, and gemm entirely.
//!   Uses a direct per-(b, c) accumulate with analytic bounds so the inner
//!   spatial loop has no padding checks and autovectorizes.
//! - Small-channel fast path: For groups=1 convs with very few input channels
//!   (e.g. 3-channel Sobel-style edge filters), reuse the depthwise kernel by
//!   accumulating over input channels. Skips the same NHWC/im2col/gemm overhead
//!   as the depthwise path. Wins when `channels_in` is small enough that
//!   gemm's per-dispatch setup cost dominates over the tiny inner compute.
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

/// Generates a conv3d typed function with 1x1, depthwise, small-channel, and
/// direct fast-path checks.
macro_rules! conv3d_typed {
    ($fn_name:ident, $T:ty, $dtype:expr, $zero:expr, $gemm_fn:ident, $add_fn:expr, $fn_1x1:ident, $fn_depthwise:ident, $fn_small_channel:ident $(, $fn_direct:ident)?) => {
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
            let x_shape = x.layout().shape();
            if should_use_depthwise_conv(x_shape, w_shape, options) {
                return $fn_depthwise(x, weight, bias, options);
            }
            if should_use_small_channel_conv(x_shape, w_shape, options) {
                return $fn_small_channel(x, weight, bias, options);
            }
            $(
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
    conv3d_depthwise_f32,
    conv3d_small_channel_f32,
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
    conv3d_depthwise_f64,
    conv3d_small_channel_f64,
    conv3d_direct_f64
);
conv3d_typed!(
    conv3d_f16,
    f16,
    DType::F16,
    f16::from_f32(0.0),
    gemm_f16,
    |a: f16, b: f16| f16::from_f32(a.to_f32() + b.to_f32()),
    conv3d_1x1_f16,
    conv3d_depthwise_f16,
    conv3d_small_channel_f16
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
// Depthwise conv fast path (no NHWC, no im2col, no gemm)
// ============================================================================
//
// For canonical depthwise convolutions where every input channel maps to
// exactly one output channel through its own filter (groups == c_in == c_out,
// channels_per_group == 1), the generic im2col + per-group gemm path pays
// enormous overhead:
//
//   - Per (tile, group) heap allocation of a col_tile buffer.
//   - Per (tile, group) gemm dispatch with M=1, K=kernel_spatial, N=tile_size.
//     The gemm kernel's setup cost dominates the tiny inner compute.
//   - Full NHWC conversion of the input even though no channel mixing occurs.
//
// The depthwise path drops all three. It walks NCHW data directly and, for
// each (batch, channel), accumulates the convolution by looping over kernel
// positions on the outside and spatial positions on the inside. The valid
// output range for each kernel position is computed analytically so the
// inner spatial loop has no padding checks and LLVM autovectorizes it in
// the common stride=1, dilation=1 case.

/// Decide whether to use the depthwise fast path.
///
/// Triggers on canonical depthwise 1D or 2D convolutions (restricted to
/// `kd == 1 && in_d == 1` after the 3D expansion used by conv1d/conv2d).
fn should_use_depthwise_conv(
    x_shape: &[usize],
    w_shape: &[usize],
    options: &ConvOptions<3>,
) -> bool {
    let channels_in = x_shape[1];
    let channels_out = w_shape[0];
    let channels_per_group = w_shape[1];
    let groups = options.groups;

    // Canonical depthwise: one input channel per group, one output channel per group.
    if channels_per_group != 1 || groups != channels_in || channels_out != channels_in {
        return false;
    }

    // Only the 1D/2D-in-3D shapes (kd=1, in_d=1) produced by the conv1d/conv2d
    // expansion. Pure 3D depthwise would need additional loop nesting.
    if w_shape[2] != 1 || x_shape[2] != 1 {
        return false;
    }
    if options.stride[0] != 1 || options.padding[0] != 0 || options.dilation[0] != 1 {
        return false;
    }

    true
}

/// Compute the half-open range `[out_start, out_end)` of output positions `o`
/// for which the corresponding input index `o * stride + k * dilation - pad`
/// lies inside `[0, in_size)`.
#[inline]
fn valid_out_range(
    k: usize,
    dilation: usize,
    pad: usize,
    stride: usize,
    in_size: usize,
    out_size: usize,
) -> (usize, usize) {
    debug_assert!(stride >= 1, "stride must be >= 1");
    let offset = k * dilation;

    // Lower: smallest o with o*stride + offset >= pad.
    let out_start = if offset >= pad {
        0
    } else {
        (pad - offset).div_ceil(stride)
    };

    // Upper (exclusive): smallest o with o*stride + offset - pad >= in_size,
    // i.e. smallest o with o*stride >= in_size + pad - offset.
    let threshold = in_size + pad;
    let out_end = if offset >= threshold {
        0
    } else {
        (threshold - offset).div_ceil(stride)
    };

    let out_end = out_end.min(out_size);
    let out_start = out_start.min(out_end);
    (out_start, out_end)
}

/// Output-plane element count above which `conv_plane_accumulate` switches
/// from the `kh, kw, oh, ow` "kh-outer" loop order to `oh, kh, kw, ow`
/// "oh-outer".
///
/// Below the threshold the whole plane fits comfortably in L1 (~32 KB on
/// most modern CPUs), so the hardware prefetcher tracks the regular
/// `oh`-stride access and the kh-outer order amortizes per-kernel-position
/// setup over long inner runs. Above the threshold the plane no longer
/// fits, and the kh-outer order refetches the output from L2/DRAM once
/// per `(kh, kw)` pair: a `kh * kw`-fold amplification of output memory
/// traffic. Flipping to oh-outer pins one output row in L1 across every
/// kernel position and traverses the plane exactly once.
///
/// 8192 f32 elements = 32 KB, i.e. L1 data-cache size on M3/M4 Max and
/// most modern x86 server parts. The gap in the conv benchmarks is clean:
/// every depthwise shape that regressed under pure oh-outer was <= 3136
/// elements, while the Sobel / preproc / mask shapes that win under
/// oh-outer are all > 65000 elements.
const CONV_PLANE_OH_OUTER_THRESHOLD: usize = 8192;

/// Accumulate one 2D conv plane: `out_plane += conv2d(in_plane, w_plane)` using
/// the precomputed analytic `oh_ranges`/`ow_ranges` to skip padding checks in
/// the inner loop.
///
/// **Precondition**: `out_plane` must already hold the running accumulator
/// (zero on the first call, or whatever partial sum has been built up across
/// previous `ci` iterations). The function reads each output element before
/// writing, so an uninitialized buffer produces silent garbage.
///
/// Dispatches to one of two loop orders based on `out_plane.len()`; see
/// `CONV_PLANE_OH_OUTER_THRESHOLD` for the tradeoff. Shared by
/// `conv3d_depthwise_impl` and `conv3d_small_channel_impl`. The dispatcher
/// is `#[inline]` (not `inline(always)`) so the runtime length branch lives
/// once at each call site instead of inlining both variants; the variants
/// themselves stay `inline(always)` so LLVM sees the concrete inner loop
/// pattern and emits SIMD fmuladd. Using `num_traits::Float` bounds (rather
/// than fn-pointer arithmetic) is load-bearing for vectorization.
#[inline]
#[allow(clippy::too_many_arguments)]
fn conv_plane_accumulate<T: num_traits::Float + Copy>(
    out_plane: &mut [T],
    in_plane: &[T],
    w_plane: &[T],
    kernel_h: usize,
    kernel_w: usize,
    in_w: usize,
    out_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    oh_ranges: &[(usize, usize)],
    ow_ranges: &[(usize, usize)],
) {
    if out_plane.len() > CONV_PLANE_OH_OUTER_THRESHOLD {
        conv_plane_accumulate_oh_outer(
            out_plane, in_plane, w_plane, kernel_h, kernel_w, in_w, out_w, stride_h, stride_w,
            pad_h, pad_w, dilation_h, dilation_w, oh_ranges, ow_ranges,
        );
    } else {
        conv_plane_accumulate_kh_outer(
            out_plane, in_plane, w_plane, kernel_h, kernel_w, in_w, out_w, stride_h, stride_w,
            pad_h, pad_w, dilation_h, dilation_w, oh_ranges, ow_ranges,
        );
    }
}

/// `oh`-outermost variant. Pins one output row in L1 across every kernel
/// position that accumulates into it, so each row is touched exactly once
/// regardless of how large the full output plane is. Selected when the
/// plane exceeds L1 (see `CONV_PLANE_OH_OUTER_THRESHOLD`).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn conv_plane_accumulate_oh_outer<T: num_traits::Float + Copy>(
    out_plane: &mut [T],
    in_plane: &[T],
    w_plane: &[T],
    kernel_h: usize,
    kernel_w: usize,
    in_w: usize,
    out_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    oh_ranges: &[(usize, usize)],
    ow_ranges: &[(usize, usize)],
) {
    // An empty plane or zero-width output is a trivial no-op. This also
    // guards the divide below against `out_w == 0`, which is not reachable
    // from the in-tree callers (burn's `calculate_conv_output_size` is
    // always >= 1 for valid inputs) but would otherwise panic if some
    // future caller handed us a degenerate slice.
    if out_plane.is_empty() || out_w == 0 {
        return;
    }

    // out_plane is a per-(batch, channel) slice of shape [out_h, out_w]
    // stored contiguously; both call sites produce it by disjoint splitting
    // of a `vec![zero; batch * channels * out_h * out_w]` allocation, so
    // the length is always an exact multiple of `out_w`. The `debug_assert`
    // turns that documentary invariant into an enforceable one.
    debug_assert_eq!(
        out_plane.len() % out_w,
        0,
        "out_plane length must be a whole number of rows"
    );
    let out_h = out_plane.len() / out_w;

    for oh in 0..out_h {
        let out_row = &mut out_plane[oh * out_w..(oh + 1) * out_w];

        for kh in 0..kernel_h {
            let (oh_start, oh_end) = oh_ranges[kh];
            // Skip kernel rows that fall outside the padded image at this oh.
            if oh < oh_start || oh >= oh_end {
                continue;
            }
            let ih = oh * stride_h + kh * dilation_h - pad_h;
            let in_row = &in_plane[ih * in_w..(ih + 1) * in_w];

            for kw in 0..kernel_w {
                let (ow_start, ow_end) = ow_ranges[kw];
                if ow_start >= ow_end {
                    continue;
                }
                let w_val = w_plane[kh * kernel_w + kw];
                // All terms are non-negative because ow_start was chosen so
                // that the corresponding `iw` is in bounds.
                let iw_start = ow_start * stride_w + kw * dilation_w - pad_w;

                if stride_w == 1 {
                    let run_len = ow_end - ow_start;
                    let in_slice = &in_row[iw_start..iw_start + run_len];
                    let out_slice = &mut out_row[ow_start..ow_end];
                    for (o, &xv) in out_slice.iter_mut().zip(in_slice.iter()) {
                        *o = *o + w_val * xv;
                    }
                } else {
                    let mut iw = iw_start;
                    for o in &mut out_row[ow_start..ow_end] {
                        *o = *o + w_val * in_row[iw];
                        iw += stride_w;
                    }
                }
            }
        }
    }
}

/// `kh, kw`-outermost variant. Amortizes per-kernel-position setup over
/// long regular `oh` runs that the hardware prefetcher can track.
/// Selected when the whole output plane already fits in L1 (see
/// `CONV_PLANE_OH_OUTER_THRESHOLD`): the oh-outer trick buys nothing
/// there and the extra per-iteration bookkeeping slightly hurts.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn conv_plane_accumulate_kh_outer<T: num_traits::Float + Copy>(
    out_plane: &mut [T],
    in_plane: &[T],
    w_plane: &[T],
    kernel_h: usize,
    kernel_w: usize,
    in_w: usize,
    out_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    oh_ranges: &[(usize, usize)],
    ow_ranges: &[(usize, usize)],
) {
    for kh in 0..kernel_h {
        let (oh_start, oh_end) = oh_ranges[kh];
        if oh_start >= oh_end {
            continue;
        }
        for kw in 0..kernel_w {
            let (ow_start, ow_end) = ow_ranges[kw];
            if ow_start >= ow_end {
                continue;
            }
            let w_val = w_plane[kh * kernel_w + kw];
            let iw_start = ow_start * stride_w + kw * dilation_w - pad_w;
            let run_len = ow_end - ow_start;
            for oh in oh_start..oh_end {
                let ih = oh * stride_h + kh * dilation_h - pad_h;
                let in_row = &in_plane[ih * in_w..(ih + 1) * in_w];
                let out_row = &mut out_plane[oh * out_w..(oh + 1) * out_w];
                if stride_w == 1 {
                    let in_slice = &in_row[iw_start..iw_start + run_len];
                    let out_slice = &mut out_row[ow_start..ow_end];
                    for (o, &xv) in out_slice.iter_mut().zip(in_slice.iter()) {
                        *o = *o + w_val * xv;
                    }
                } else {
                    let mut iw = iw_start;
                    for o in &mut out_row[ow_start..ow_end] {
                        *o = *o + w_val * in_row[iw];
                        iw += stride_w;
                    }
                }
            }
        }
    }
}

macro_rules! conv3d_depthwise_typed {
    ($fn_name:ident, $T:ty, $dtype:expr) => {
        fn $fn_name(
            x: FlexTensor,
            weight: FlexTensor,
            bias: Option<FlexTensor>,
            options: &ConvOptions<3>,
        ) -> FlexTensor {
            conv3d_depthwise_impl::<$T>(x, weight, bias, options, $dtype)
        }
    };
}

conv3d_depthwise_typed!(conv3d_depthwise_f32, f32, DType::F32);
conv3d_depthwise_typed!(conv3d_depthwise_f64, f64, DType::F64);
conv3d_depthwise_typed!(conv3d_depthwise_f16, f16, DType::F16);

/// Depthwise conv3d: one filter per channel, no channel mixing.
///
/// Preconditions (checked by `should_use_depthwise_conv`):
/// - `channels_per_group == 1`
/// - `groups == channels_in == channels_out`
/// - `kernel_d == 1`, `in_d == 1`, trivial d-axis options.
///
/// Uses `num_traits::Float` bounds so the inner multiply-accumulate compiles
/// down to direct `fmul`/`fadd` instructions that LLVM can autovectorize.
/// Function-pointer arithmetic (`fn(T, T) -> T`) would prevent vectorization
/// because each call is an indirect branch that blocks the loop pattern match.
fn conv3d_depthwise_impl<T>(
    x: FlexTensor,
    weight: FlexTensor,
    bias: Option<FlexTensor>,
    options: &ConvOptions<3>,
    dtype: DType,
) -> FlexTensor
where
    T: num_traits::Float + bytemuck::Pod + Clone + Copy + burn_backend::Element + Send + Sync,
{
    let zero = <T as num_traits::Zero>::zero();
    let x = x.to_contiguous();
    let weight = weight.to_contiguous();

    let x_shape = x.layout().shape();
    let w_shape = weight.layout().shape();

    let batch_size = x_shape[0];
    let channels = x_shape[1];
    let in_h = x_shape[3];
    let in_w = x_shape[4];

    let kernel_h = w_shape[3];
    let kernel_w = w_shape[4];

    let [_, stride_h, stride_w] = options.stride;
    let [_, pad_h, pad_w] = options.padding;
    let [_, dilation_h, dilation_w] = options.dilation;

    let out_h = calculate_conv_output_size(kernel_h, stride_h, pad_h, dilation_h, in_h);
    let out_w = calculate_conv_output_size(kernel_w, stride_w, pad_w, dilation_w, in_w);

    let total = [batch_size, channels, out_h, out_w]
        .iter()
        .try_fold(1usize, |acc, &x| acc.checked_mul(x))
        .expect("conv depthwise: output dimensions would overflow");

    let x_data: &[T] = x.storage();
    let w_data: &[T] = weight.storage();

    let in_spatial = in_h * in_w;
    let out_spatial = out_h * out_w;
    let k_spatial = kernel_h * kernel_w;

    // Valid oh/ow ranges are identical for every (b, c), so precompute once.
    let oh_ranges: Vec<(usize, usize)> = (0..kernel_h)
        .map(|kh| valid_out_range(kh, dilation_h, pad_h, stride_h, in_h, out_h))
        .collect();
    let ow_ranges: Vec<(usize, usize)> = (0..kernel_w)
        .map(|kw| valid_out_range(kw, dilation_w, pad_w, stride_w, in_w, out_w))
        .collect();

    let mut output = vec![zero; total];

    // Per-plane work for one `(batch, channel)` pair. Output slice is passed
    // in so the rayon and sequential dispatch paths can source it differently
    // (from a raw pointer or a direct borrow) without duplicating this body.
    let plane_work = |bc: usize, out_plane: &mut [T]| {
        let c = bc % channels;
        let in_base = bc * in_spatial;
        let w_base = c * k_spatial;
        conv_plane_accumulate(
            out_plane,
            &x_data[in_base..in_base + in_spatial],
            &w_data[w_base..w_base + k_spatial],
            kernel_h,
            kernel_w,
            in_w,
            out_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            &oh_ranges,
            &ow_ranges,
        );
    };

    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;

        let dst_ptr = crate::ops::SendMutPtr::new(output.as_mut_ptr());
        (0..batch_size * channels).into_par_iter().for_each(|bc| {
            // SAFETY: disjoint `[bc * out_spatial, (bc+1) * out_spatial)`
            // ranges tile the output buffer.
            let out_plane: &mut [T] = unsafe {
                core::slice::from_raw_parts_mut(dst_ptr.ptr_add(bc * out_spatial), out_spatial)
            };
            plane_work(bc, out_plane);
        });
    }
    #[cfg(not(feature = "rayon"))]
    {
        for bc in 0..batch_size * channels {
            let out_base = bc * out_spatial;
            plane_work(bc, &mut output[out_base..out_base + out_spatial]);
        }
    }

    if let Some(bias) = bias {
        let bias = bias.to_contiguous();
        let bias_data: &[T] = bias.storage();
        assert_eq!(
            bias_data.len(),
            channels,
            "conv depthwise: bias length ({}) must equal channels ({channels})",
            bias_data.len()
        );
        add_bias(
            &mut output,
            bias_data,
            batch_size,
            channels,
            out_spatial,
            |a, b| a + b,
        );
    }

    let out_shape = Shape::from(vec![batch_size, channels, 1, out_h, out_w]);
    FlexTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(out_shape),
        dtype,
    )
}

// ============================================================================
// Small-channel conv fast path (no NHWC, no im2col, no gemm)
// ============================================================================
//
// For `groups=1` convs with very few input channels (e.g. 3-channel Sobel
// filters, single-channel mask networks, early-stage image preprocessors),
// the generic `conv3d_impl` pays the same kind of overhead as the depthwise
// case: gemm is dispatched with `M = channels_out, K = channels_in * k_spatial,
// N = tile_size`, and at small `K` the gemm kernel's pack + select + dispatch
// cost dominates the actual FMAs. On top of that, the full NHWC conversion
// of the input is pure memory traffic that buys nothing when the channel
// dimension is already tiny.
//
// The small-channel path walks NCHW data directly. For each `(batch, out_ch)`
// pair it accumulates contributions from every input channel by calling the
// same `conv_plane_accumulate` helper used by the depthwise path. Compared
// to the depthwise case this adds an outer `for ci in 0..channels_in` loop
// around the accumulate; everything else (analytic output ranges, parallel
// fan-out, bias add) is identical.

/// Threshold on `channels_in` for the small-channel fast path.
///
/// Picked empirically: at `channels_in <= 4`, gemm dispatch overhead exceeds
/// the inner compute by a clear margin. Tuning higher risks regressing shapes
/// where gemm's data reuse across output channels wins.
const SMALL_CHANNEL_IN_THRESHOLD: usize = 4;

/// Threshold on `channels_out` for the small-channel fast path.
///
/// The small-channel path loops `for co: for ci:` and re-reads each input
/// channel once per output channel. At large `channels_out`, that redundant
/// memory traffic dominates over gemm's register-tiled data reuse. Picked
/// empirically: classic ImageNet first-layer (`3 -> 64`) runs ~5x slower on
/// the direct path than on gemm, so we cap at 16 to stay safely on the right
/// side of that cliff. Sobel-style filters (`3 -> 3..8`) are well under this
/// and keep their 1.1-1.4x win over burn-ndarray.
const SMALL_CHANNEL_OUT_THRESHOLD: usize = 16;

/// Decide whether to use the small-channel fast path.
///
/// Triggers on `groups=1` 2D-in-3D convs (or 1D via the conv1d expansion)
/// with both small `channels_in` and small `channels_out`. Non-depthwise
/// grouped convs and the many-channel case both go through the generic
/// `conv3d_impl`.
fn should_use_small_channel_conv(
    x_shape: &[usize],
    w_shape: &[usize],
    options: &ConvOptions<3>,
) -> bool {
    if options.groups != 1 {
        return false;
    }

    // Only the 1D/2D-in-3D shapes (kd=1, in_d=1). Pure 3D convs do not benefit
    // and would require adding a d-axis loop.
    if w_shape[2] != 1 || x_shape[2] != 1 {
        return false;
    }
    if options.stride[0] != 1 || options.padding[0] != 0 || options.dilation[0] != 1 {
        return false;
    }

    let channels_in = x_shape[1];
    let channels_out = w_shape[0];
    channels_in > 0
        && channels_in <= SMALL_CHANNEL_IN_THRESHOLD
        && channels_out > 0
        && channels_out <= SMALL_CHANNEL_OUT_THRESHOLD
}

macro_rules! conv3d_small_channel_typed {
    ($fn_name:ident, $T:ty, $dtype:expr) => {
        fn $fn_name(
            x: FlexTensor,
            weight: FlexTensor,
            bias: Option<FlexTensor>,
            options: &ConvOptions<3>,
        ) -> FlexTensor {
            conv3d_small_channel_impl::<$T>(x, weight, bias, options, $dtype)
        }
    };
}

conv3d_small_channel_typed!(conv3d_small_channel_f32, f32, DType::F32);
conv3d_small_channel_typed!(conv3d_small_channel_f64, f64, DType::F64);
conv3d_small_channel_typed!(conv3d_small_channel_f16, f16, DType::F16);

/// Small-channel conv3d: `groups=1` with small `channels_in`.
///
/// Preconditions (checked by `should_use_small_channel_conv`):
/// - `options.groups == 1`
/// - `channels_in <= SMALL_CHANNEL_IN_THRESHOLD`
/// - `kernel_d == 1`, `in_d == 1`, trivial d-axis options.
fn conv3d_small_channel_impl<T>(
    x: FlexTensor,
    weight: FlexTensor,
    bias: Option<FlexTensor>,
    options: &ConvOptions<3>,
    dtype: DType,
) -> FlexTensor
where
    T: num_traits::Float + bytemuck::Pod + Clone + Copy + burn_backend::Element + Send + Sync,
{
    let zero = <T as num_traits::Zero>::zero();
    let x = x.to_contiguous();
    let weight = weight.to_contiguous();

    let x_shape = x.layout().shape();
    let w_shape = weight.layout().shape();

    let batch_size = x_shape[0];
    let channels_in = x_shape[1];
    let in_h = x_shape[3];
    let in_w = x_shape[4];

    let channels_out = w_shape[0];
    let kernel_h = w_shape[3];
    let kernel_w = w_shape[4];

    let [_, stride_h, stride_w] = options.stride;
    let [_, pad_h, pad_w] = options.padding;
    let [_, dilation_h, dilation_w] = options.dilation;

    let out_h = calculate_conv_output_size(kernel_h, stride_h, pad_h, dilation_h, in_h);
    let out_w = calculate_conv_output_size(kernel_w, stride_w, pad_w, dilation_w, in_w);

    let total = [batch_size, channels_out, out_h, out_w]
        .iter()
        .try_fold(1usize, |acc, &x| acc.checked_mul(x))
        .expect("conv small-channel: output dimensions would overflow");

    let x_data: &[T] = x.storage();
    let w_data: &[T] = weight.storage();

    let in_spatial = in_h * in_w;
    let out_spatial = out_h * out_w;
    let k_spatial = kernel_h * kernel_w;
    let w_co_stride = channels_in * k_spatial;
    let x_batch_stride = channels_in * in_spatial;

    let oh_ranges: Vec<(usize, usize)> = (0..kernel_h)
        .map(|kh| valid_out_range(kh, dilation_h, pad_h, stride_h, in_h, out_h))
        .collect();
    let ow_ranges: Vec<(usize, usize)> = (0..kernel_w)
        .map(|kw| valid_out_range(kw, dilation_w, pad_w, stride_w, in_w, out_w))
        .collect();

    let mut output = vec![zero; total];

    // Per-plane work for one `(batch, out_channel)` pair. Accumulates
    // `sum over ci of conv2d(in[b, ci], w[co, ci])` into `out_plane` by
    // calling the shared helper once per input channel.
    let plane_work = |b_co: usize, out_plane: &mut [T]| {
        let b = b_co / channels_out;
        let co = b_co % channels_out;
        for ci in 0..channels_in {
            let in_base = b * x_batch_stride + ci * in_spatial;
            let w_base = co * w_co_stride + ci * k_spatial;
            conv_plane_accumulate(
                out_plane,
                &x_data[in_base..in_base + in_spatial],
                &w_data[w_base..w_base + k_spatial],
                kernel_h,
                kernel_w,
                in_w,
                out_w,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                dilation_h,
                dilation_w,
                &oh_ranges,
                &ow_ranges,
            );
        }
    };

    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;

        let dst_ptr = crate::ops::SendMutPtr::new(output.as_mut_ptr());
        (0..batch_size * channels_out)
            .into_par_iter()
            .for_each(|b_co| {
                // SAFETY: disjoint `[b_co * out_spatial, (b_co+1) * out_spatial)`
                // ranges tile the output buffer.
                let out_plane: &mut [T] = unsafe {
                    core::slice::from_raw_parts_mut(
                        dst_ptr.ptr_add(b_co * out_spatial),
                        out_spatial,
                    )
                };
                plane_work(b_co, out_plane);
            });
    }
    #[cfg(not(feature = "rayon"))]
    {
        for b_co in 0..batch_size * channels_out {
            let out_base = b_co * out_spatial;
            plane_work(b_co, &mut output[out_base..out_base + out_spatial]);
        }
    }

    if let Some(bias) = bias {
        let bias = bias.to_contiguous();
        let bias_data: &[T] = bias.storage();
        assert_eq!(
            bias_data.len(),
            channels_out,
            "conv small-channel: bias length ({}) must equal channels_out ({channels_out})",
            bias_data.len()
        );
        add_bias(
            &mut output,
            bias_data,
            batch_size,
            channels_out,
            out_spatial,
            |a, b| a + b,
        );
    }

    let out_shape = Shape::from(vec![batch_size, channels_out, 1, out_h, out_w]);
    FlexTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(out_shape),
        dtype,
    )
}

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

    // ========================================================================
    // Depthwise conv fast-path tests
    // ========================================================================
    //
    // These tests exercise `conv3d_depthwise_impl` and cross-check against a
    // naive reference to catch any indexing, bounds, or accumulation mistakes.

    /// Naive NCHW depthwise conv2d reference implementation.
    /// Returns output with shape `[batch, channels, out_h, out_w]`.
    #[allow(clippy::too_many_arguments)]
    fn naive_depthwise_conv2d_f32(
        x: &[f32],
        w: &[f32],
        bias: Option<&[f32]>,
        batch: usize,
        channels: usize,
        in_h: usize,
        in_w: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
        dilation_h: usize,
        dilation_w: usize,
    ) -> (Vec<f32>, usize, usize) {
        let out_h = (in_h + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
        let out_w = (in_w + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
        let mut out = vec![0.0f32; batch * channels * out_h * out_w];
        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut acc = 0.0f32;
                        for kh in 0..kernel_h {
                            let ih = oh as isize * stride_h as isize
                                + kh as isize * dilation_h as isize
                                - pad_h as isize;
                            if ih < 0 || ih >= in_h as isize {
                                continue;
                            }
                            for kw in 0..kernel_w {
                                let iw = ow as isize * stride_w as isize
                                    + kw as isize * dilation_w as isize
                                    - pad_w as isize;
                                if iw < 0 || iw >= in_w as isize {
                                    continue;
                                }
                                let x_idx =
                                    ((b * channels + c) * in_h + ih as usize) * in_w + iw as usize;
                                let w_idx = (c * kernel_h + kh) * kernel_w + kw;
                                acc += x[x_idx] * w[w_idx];
                            }
                        }
                        if let Some(bias) = bias {
                            acc += bias[c];
                        }
                        let o_idx = ((b * channels + c) * out_h + oh) * out_w + ow;
                        out[o_idx] = acc;
                    }
                }
            }
        }
        (out, out_h, out_w)
    }

    /// Build deterministic pseudo-random input for a depthwise test.
    fn seeded_vec_f32(n: usize, seed: u32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let v = ((i as u32).wrapping_mul(2654435761).wrapping_add(seed)) & 0xffff;
                (v as f32 / 32768.0) - 1.0
            })
            .collect()
    }

    /// Shared helper: run conv2d via the public dispatch (which must pick the
    /// depthwise path for these shapes) and compare to the naive reference.
    #[allow(clippy::too_many_arguments)]
    fn check_depthwise_conv2d_f32(
        batch: usize,
        channels: usize,
        in_h: usize,
        in_w: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        with_bias: bool,
    ) {
        let x_vec = seeded_vec_f32(batch * channels * in_h * in_w, 1);
        let w_vec = seeded_vec_f32(channels * kernel_h * kernel_w, 2);
        let bias_vec = if with_bias {
            Some(seeded_vec_f32(channels, 3))
        } else {
            None
        };

        let (expected, out_h, out_w) = naive_depthwise_conv2d_f32(
            &x_vec,
            &w_vec,
            bias_vec.as_deref(),
            batch,
            channels,
            in_h,
            in_w,
            kernel_h,
            kernel_w,
            stride[0],
            stride[1],
            padding[0],
            padding[1],
            dilation[0],
            dilation[1],
        );

        let x = FlexTensor::from_data(TensorData::new(x_vec, vec![batch, channels, in_h, in_w]));
        let weight = FlexTensor::from_data(TensorData::new(
            w_vec,
            vec![channels, 1, kernel_h, kernel_w],
        ));
        let bias = bias_vec.map(|v| FlexTensor::from_data(TensorData::new(v, vec![channels])));
        let options = ConvOptions::new(stride, padding, dilation, channels);
        let result = conv2d_f32(x, weight, bias, &options);

        assert_eq!(
            result.layout().shape().to_vec(),
            vec![batch, channels, out_h, out_w],
            "output shape mismatch"
        );

        let out: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(out.len(), expected.len());
        for (i, (a, e)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-4,
                "mismatch at {i}: got {a}, expected {e}"
            );
        }
    }

    #[test]
    fn test_conv2d_depthwise_3x3_no_pad() {
        // Canonical depthwise: groups == channels_in == channels_out = 8, 3x3.
        check_depthwise_conv2d_f32(2, 8, 16, 16, 3, 3, [1, 1], [0, 0], [1, 1], false);
    }

    #[test]
    fn test_conv2d_depthwise_3x3_pad1() {
        // Same input and output size via padding=1.
        check_depthwise_conv2d_f32(2, 8, 16, 16, 3, 3, [1, 1], [1, 1], [1, 1], false);
    }

    #[test]
    fn test_conv2d_depthwise_3x3_stride2_pad1() {
        // Halve spatial via stride 2, with padding.
        check_depthwise_conv2d_f32(1, 16, 32, 32, 3, 3, [2, 2], [1, 1], [1, 1], false);
    }

    #[test]
    fn test_conv2d_depthwise_5x5_pad2() {
        check_depthwise_conv2d_f32(2, 4, 10, 10, 5, 5, [1, 1], [2, 2], [1, 1], false);
    }

    #[test]
    fn test_conv2d_depthwise_7x7_pad3() {
        // Large kernel like ConvNeXt's 7x7 depthwise.
        check_depthwise_conv2d_f32(2, 24, 14, 14, 7, 7, [1, 1], [3, 3], [1, 1], false);
    }

    #[test]
    fn test_conv2d_depthwise_dilated() {
        // Dilation 2 means the effective receptive field is 5x5 but with gaps.
        check_depthwise_conv2d_f32(1, 8, 12, 12, 3, 3, [1, 1], [2, 2], [2, 2], false);
    }

    #[test]
    fn test_conv2d_depthwise_with_bias() {
        check_depthwise_conv2d_f32(2, 8, 8, 8, 3, 3, [1, 1], [1, 1], [1, 1], true);
    }

    #[test]
    fn test_conv2d_depthwise_single_channel() {
        // groups = channels_in = 1 is a degenerate depthwise which should
        // still go through this path (it also looks identical to an
        // ungrouped 1-channel conv, but validates the range math).
        check_depthwise_conv2d_f32(1, 1, 5, 5, 3, 3, [1, 1], [1, 1], [1, 1], false);
    }

    #[test]
    fn test_conv2d_depthwise_asymmetric_kernel() {
        // Non-square kernel.
        check_depthwise_conv2d_f32(1, 4, 8, 12, 3, 5, [1, 1], [1, 2], [1, 1], false);
    }

    #[test]
    fn test_conv2d_depthwise_f64() {
        // Smoke test f64 dispatch through depthwise path.
        let x_data: Vec<f64> = (0..2 * 4 * 5 * 5).map(|i| (i as f64) * 0.1).collect();
        let w_data: Vec<f64> = (0..4 * 3 * 3).map(|i| (i as f64) * 0.01).collect();
        let x = FlexTensor::from_data(TensorData::new(x_data.clone(), vec![2, 4, 5, 5]));
        let weight = FlexTensor::from_data(TensorData::new(w_data.clone(), vec![4, 1, 3, 3]));
        let options = ConvOptions::new([1, 1], [1, 1], [1, 1], 4);
        let result = conv2d_f64(x, weight, None, &options);
        let out: Vec<f64> = result.into_data().to_vec().unwrap();

        // Verify against a naive f64 reference for one element (center of channel 2).
        let b = 0usize;
        let c = 2usize;
        let oh = 2usize;
        let ow = 2usize;
        let mut expected = 0.0f64;
        for kh in 0..3 {
            for kw in 0..3 {
                let ih = oh as isize + kh as isize - 1;
                let iw = ow as isize + kw as isize - 1;
                if ih >= 0 && ih < 5 && iw >= 0 && iw < 5 {
                    let x_idx = ((b * 4 + c) * 5 + ih as usize) * 5 + iw as usize;
                    let w_idx = (c * 3 + kh) * 3 + kw;
                    expected += x_data[x_idx] * w_data[w_idx];
                }
            }
        }
        let out_idx = ((b * 4 + c) * 5 + oh) * 5 + ow;
        assert!((out[out_idx] - expected).abs() < 1e-10);
    }

    #[test]
    fn test_conv2d_depthwise_f16() {
        // Validate f16 depthwise dispatch against a naive f32 reference.
        // Shape check alone would miss indexing/accumulation bugs that still
        // happen to produce the right shape.
        use burn_std::f16;
        let x_data_f32: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let w_data_f32: Vec<f32> = (0..16).map(|i| i as f32 * 0.01).collect();
        let x_data: Vec<f16> = x_data_f32.iter().copied().map(f16::from_f32).collect();
        let w_data: Vec<f16> = w_data_f32.iter().copied().map(f16::from_f32).collect();
        let x = FlexTensor::from_data(TensorData::new(x_data, vec![1, 4, 2, 2]));
        let weight = FlexTensor::from_data(TensorData::new(w_data, vec![4, 1, 2, 2]));
        let options = ConvOptions::new([1, 1], [0, 0], [1, 1], 4);
        let result = conv2d_f16(x, weight, None, &options);
        assert_eq!(result.layout().shape().to_vec(), vec![1, 4, 1, 1]);
        let out: Vec<f16> = result.into_data().to_vec().unwrap();

        // Depthwise: out[c] = sum over (kh, kw) of x[c, kh, kw] * w[c, 0, kh, kw].
        // The input per-channel is 4 elements (2x2) and the kernel is 2x2, so
        // there's exactly one output pixel per channel.
        for c in 0..4 {
            let mut expected = 0.0f32;
            for k in 0..4 {
                // c * 4 indexes into channel c's 2x2 plane; both x and w share
                // the same [c, ...] offset because the depthwise weight layout
                // is [c_out, 1, kh, kw].
                expected += x_data_f32[c * 4 + k] * w_data_f32[c * 4 + k];
            }
            let actual = out[c].to_f32();
            assert!(
                (actual - expected).abs() < 1e-2,
                "f16 depthwise mismatch at c={c}: expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn test_conv1d_depthwise() {
        // conv1d -> conv3d expansion (kd=kh=1), depthwise on the last axis.
        let channels = 4;
        let in_w = 16;
        let kw = 3;
        let x_data = seeded_vec_f32(channels * in_w, 10);
        let w_data = seeded_vec_f32(channels * kw, 20);
        let x = FlexTensor::from_data(TensorData::new(x_data.clone(), vec![1, channels, in_w]));
        let weight = FlexTensor::from_data(TensorData::new(w_data.clone(), vec![channels, 1, kw]));
        let options = ConvOptions::new([1], [1], [1], channels);
        let result = conv1d_f32(x, weight, None, &options);
        let out_w = in_w;
        assert_eq!(result.layout().shape().to_vec(), vec![1, channels, out_w]);
        let out: Vec<f32> = result.into_data().to_vec().unwrap();

        // Naive reference.
        for c in 0..channels {
            for o in 0..out_w {
                let mut expected = 0.0f32;
                for k in 0..kw {
                    let i = o as isize + k as isize - 1;
                    if i >= 0 && i < in_w as isize {
                        expected += x_data[c * in_w + i as usize] * w_data[c * kw + k];
                    }
                }
                let actual = out[c * out_w + o];
                assert!(
                    (actual - expected).abs() < 1e-5,
                    "conv1d depthwise mismatch at c={c}, o={o}: expected {expected}, got {actual}"
                );
            }
        }
    }

    #[test]
    fn test_conv1d_depthwise_stride_batch_bias() {
        // Depthwise conv1d with batch > 1, stride > 1, padding, and bias.
        // conv1d flows through the same conv3d_depthwise_impl as conv2d
        // because conv1d expands to a 3D shape with kd=kh=1. This test
        // validates the full dispatch path with a non-trivial stride.
        let batch = 3;
        let channels = 6;
        let in_w = 32;
        let kw = 5;
        let stride = 2;
        let pad = 2;
        let out_w = (in_w + 2 * pad - kw) / stride + 1;

        let x_data = seeded_vec_f32(batch * channels * in_w, 30);
        let w_data = seeded_vec_f32(channels * kw, 40);
        let bias_data = seeded_vec_f32(channels, 50);

        let x = FlexTensor::from_data(TensorData::new(x_data.clone(), vec![batch, channels, in_w]));
        let weight = FlexTensor::from_data(TensorData::new(w_data.clone(), vec![channels, 1, kw]));
        let bias = FlexTensor::from_data(TensorData::new(bias_data.clone(), vec![channels]));
        let options = ConvOptions::new([stride], [pad], [1], channels);
        let result = conv1d_f32(x, weight, Some(bias), &options);
        assert_eq!(
            result.layout().shape().to_vec(),
            vec![batch, channels, out_w]
        );
        let out: Vec<f32> = result.into_data().to_vec().unwrap();

        // Naive reference.
        for b in 0..batch {
            for c in 0..channels {
                for o in 0..out_w {
                    let mut expected = bias_data[c];
                    for k in 0..kw {
                        let i = (o as isize * stride as isize) + k as isize - pad as isize;
                        if i >= 0 && i < in_w as isize {
                            let x_idx = (b * channels + c) * in_w + i as usize;
                            let w_idx = c * kw + k;
                            expected += x_data[x_idx] * w_data[w_idx];
                        }
                    }
                    let actual = out[(b * channels + c) * out_w + o];
                    assert!(
                        (actual - expected).abs() < 1e-4,
                        "conv1d depthwise mismatch at b={b}, c={c}, o={o}: expected {expected}, got {actual}"
                    );
                }
            }
        }
    }

    // ========================================================================
    // Small-channel conv fast-path tests
    // ========================================================================
    //
    // These tests exercise `conv3d_small_channel_impl` for groups=1 convs
    // with `channels_in <= SMALL_CHANNEL_IN_THRESHOLD`. They cross-check the
    // output against a naive NCHW reference that sums over both input
    // channels and kernel positions.

    #[allow(clippy::too_many_arguments)]
    fn naive_conv2d_f32(
        x: &[f32],
        w: &[f32],
        bias: Option<&[f32]>,
        batch: usize,
        channels_in: usize,
        channels_out: usize,
        in_h: usize,
        in_w: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
        dilation_h: usize,
        dilation_w: usize,
    ) -> (Vec<f32>, usize, usize) {
        let out_h = (in_h + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
        let out_w = (in_w + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
        let mut out = vec![0.0f32; batch * channels_out * out_h * out_w];
        for b in 0..batch {
            for co in 0..channels_out {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut acc = 0.0f32;
                        for ci in 0..channels_in {
                            for kh in 0..kernel_h {
                                let ih = oh as isize * stride_h as isize
                                    + kh as isize * dilation_h as isize
                                    - pad_h as isize;
                                if ih < 0 || ih >= in_h as isize {
                                    continue;
                                }
                                for kw in 0..kernel_w {
                                    let iw = ow as isize * stride_w as isize
                                        + kw as isize * dilation_w as isize
                                        - pad_w as isize;
                                    if iw < 0 || iw >= in_w as isize {
                                        continue;
                                    }
                                    let x_idx = ((b * channels_in + ci) * in_h + ih as usize)
                                        * in_w
                                        + iw as usize;
                                    let w_idx =
                                        ((co * channels_in + ci) * kernel_h + kh) * kernel_w + kw;
                                    acc += x[x_idx] * w[w_idx];
                                }
                            }
                        }
                        if let Some(bias) = bias {
                            acc += bias[co];
                        }
                        let o_idx = ((b * channels_out + co) * out_h + oh) * out_w + ow;
                        out[o_idx] = acc;
                    }
                }
            }
        }
        (out, out_h, out_w)
    }

    #[allow(clippy::too_many_arguments)]
    fn check_small_channel_conv2d_f32(
        batch: usize,
        channels_in: usize,
        channels_out: usize,
        in_h: usize,
        in_w: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        with_bias: bool,
    ) {
        let x_vec = seeded_vec_f32(batch * channels_in * in_h * in_w, 100);
        let w_vec = seeded_vec_f32(channels_out * channels_in * kernel_h * kernel_w, 200);
        let bias_vec = if with_bias {
            Some(seeded_vec_f32(channels_out, 300))
        } else {
            None
        };

        let (expected, out_h, out_w) = naive_conv2d_f32(
            &x_vec,
            &w_vec,
            bias_vec.as_deref(),
            batch,
            channels_in,
            channels_out,
            in_h,
            in_w,
            kernel_h,
            kernel_w,
            stride[0],
            stride[1],
            padding[0],
            padding[1],
            dilation[0],
            dilation[1],
        );

        let x = FlexTensor::from_data(TensorData::new(x_vec, vec![batch, channels_in, in_h, in_w]));
        let weight = FlexTensor::from_data(TensorData::new(
            w_vec,
            vec![channels_out, channels_in, kernel_h, kernel_w],
        ));
        let bias = bias_vec.map(|v| FlexTensor::from_data(TensorData::new(v, vec![channels_out])));
        let options = ConvOptions::new(stride, padding, dilation, 1);
        let result = conv2d_f32(x, weight, bias, &options);

        assert_eq!(
            result.layout().shape().to_vec(),
            vec![batch, channels_out, out_h, out_w],
            "output shape mismatch"
        );

        let out: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(out.len(), expected.len());
        for (i, (a, e)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-3,
                "mismatch at {i}: got {a}, expected {e}"
            );
        }
    }

    #[test]
    fn test_conv2d_small_channel_3in_8out_k3x3_pad1() {
        // Sobel-like: 3 input channels, several output channels, 3x3 kernel.
        check_small_channel_conv2d_f32(2, 3, 8, 16, 16, 3, 3, [1, 1], [1, 1], [1, 1], false);
    }

    #[test]
    fn test_conv2d_small_channel_3in_3out_k3x3_no_pad() {
        // Channels_in == channels_out == 3, same count but groups=1 so each
        // output channel combines all input channels (not depthwise).
        check_small_channel_conv2d_f32(1, 3, 3, 10, 10, 3, 3, [1, 1], [0, 0], [1, 1], false);
    }

    #[test]
    fn test_conv2d_small_channel_3in_16out_k5x5_pad2() {
        check_small_channel_conv2d_f32(2, 3, 16, 12, 12, 5, 5, [1, 1], [2, 2], [1, 1], false);
    }

    #[test]
    fn test_conv2d_small_channel_4in_8out_k3x3_stride2() {
        // Threshold channels_in == 4.
        check_small_channel_conv2d_f32(1, 4, 8, 16, 16, 3, 3, [2, 2], [1, 1], [1, 1], false);
    }

    #[test]
    fn test_conv2d_small_channel_2in_4out_dilated() {
        check_small_channel_conv2d_f32(1, 2, 4, 12, 12, 3, 3, [1, 1], [2, 2], [2, 2], false);
    }

    #[test]
    fn test_conv2d_small_channel_with_bias() {
        check_small_channel_conv2d_f32(2, 3, 8, 8, 8, 3, 3, [1, 1], [1, 1], [1, 1], true);
    }

    #[test]
    fn test_conv2d_small_channel_asymmetric_kernel() {
        // 1x3 and 3x1 kernels are common for separable edge filters.
        check_small_channel_conv2d_f32(1, 3, 6, 16, 16, 1, 3, [1, 1], [0, 1], [1, 1], false);
        check_small_channel_conv2d_f32(1, 3, 6, 16, 16, 3, 1, [1, 1], [1, 0], [1, 1], false);
    }

    #[test]
    fn test_conv2d_small_channel_f64() {
        // Smoke test f64 dispatch.
        let x_data: Vec<f64> = (0..2 * 3 * 5 * 5).map(|i| (i as f64) * 0.1).collect();
        let w_data: Vec<f64> = (0..4 * 3 * 3 * 3).map(|i| (i as f64) * 0.01).collect();
        let x = FlexTensor::from_data(TensorData::new(x_data.clone(), vec![2, 3, 5, 5]));
        let weight = FlexTensor::from_data(TensorData::new(w_data.clone(), vec![4, 3, 3, 3]));
        let options = ConvOptions::new([1, 1], [1, 1], [1, 1], 1);
        let result = conv2d_f64(x, weight, None, &options);
        let out: Vec<f64> = result.into_data().to_vec().unwrap();

        // Verify against a naive reference for one element (center of channel 2).
        let b = 0usize;
        let co = 2usize;
        let oh = 2usize;
        let ow = 2usize;
        let mut expected = 0.0f64;
        for ci in 0..3 {
            for kh in 0..3 {
                for kw in 0..3 {
                    let ih = oh as isize + kh as isize - 1;
                    let iw = ow as isize + kw as isize - 1;
                    if ih >= 0 && ih < 5 && iw >= 0 && iw < 5 {
                        let x_idx = ((b * 3 + ci) * 5 + ih as usize) * 5 + iw as usize;
                        let w_idx = ((co * 3 + ci) * 3 + kh) * 3 + kw;
                        expected += x_data[x_idx] * w_data[w_idx];
                    }
                }
            }
        }
        let out_idx = ((b * 4 + co) * 5 + oh) * 5 + ow;
        assert!(
            (out[out_idx] - expected).abs() < 1e-10,
            "got {}, expected {expected}",
            out[out_idx]
        );
    }

    #[test]
    fn test_conv2d_small_channel_f16() {
        // Validate f16 dispatch through the small-channel path by running
        // the same shape through the f32 path and comparing element-wise.
        // This catches indexing/accumulation bugs in addition to regressions
        // in the `num_traits::Float` monomorphization for f16.
        use burn_std::f16;
        let x_data_f32: Vec<f32> = (0..3 * 4 * 4).map(|i| i as f32 * 0.1).collect();
        let w_data_f32: Vec<f32> = (0..4 * 3 * 3 * 3).map(|i| i as f32 * 0.01).collect();

        let x_data_f16: Vec<f16> = x_data_f32.iter().copied().map(f16::from_f32).collect();
        let w_data_f16: Vec<f16> = w_data_f32.iter().copied().map(f16::from_f32).collect();

        let x_f16 = FlexTensor::from_data(TensorData::new(x_data_f16, vec![1, 3, 4, 4]));
        let weight_f16 = FlexTensor::from_data(TensorData::new(w_data_f16, vec![4, 3, 3, 3]));
        let x_f32 = FlexTensor::from_data(TensorData::new(x_data_f32, vec![1, 3, 4, 4]));
        let weight_f32 = FlexTensor::from_data(TensorData::new(w_data_f32, vec![4, 3, 3, 3]));

        let options = ConvOptions::new([1, 1], [1, 1], [1, 1], 1);
        let result_f16 = conv2d_f16(x_f16, weight_f16, None, &options);
        let result_f32 = conv2d_f32(x_f32, weight_f32, None, &options);
        assert_eq!(result_f16.layout().shape().to_vec(), vec![1, 4, 4, 4]);
        assert_eq!(result_f32.layout().shape().to_vec(), vec![1, 4, 4, 4]);

        let out_f16: Vec<f16> = result_f16.into_data().to_vec().unwrap();
        let out_f32: Vec<f32> = result_f32.into_data().to_vec().unwrap();
        assert_eq!(out_f16.len(), out_f32.len());

        // f16 has ~11 bits of mantissa (~0.1% relative precision). With
        // accumulation across `c_in * k_spatial = 27` FMAs the relative
        // error can grow to a few times that. Use a relative tolerance
        // scaled by the expected magnitude with a small absolute floor for
        // values near zero.
        let rel_tol = 3e-3f32;
        let abs_tol = 1e-2f32;
        for (i, (actual, expected)) in out_f16.iter().zip(out_f32.iter()).enumerate() {
            let actual_f32 = actual.to_f32();
            let bound = (expected.abs() * rel_tol).max(abs_tol);
            assert!(
                !actual_f32.is_nan() && (actual_f32 - expected).abs() <= bound,
                "f16 small-channel mismatch at {i}: got {actual_f32}, expected {expected}, bound {bound}"
            );
        }
    }

    #[test]
    fn test_conv2d_small_channel_bias_length_mismatch_panics() {
        // The small-channel impl must panic loudly when bias length does not
        // match channels_out. A silent truncation here would make models with
        // misconfigured bias silently produce wrong output.
        let x = FlexTensor::from_data(TensorData::new(vec![0.0f32; 48], vec![1, 3, 4, 4]));
        let weight = FlexTensor::from_data(TensorData::new(vec![0.0f32; 108], vec![4, 3, 3, 3]));
        // Bias has only 2 elements but there are 4 output channels.
        let bias = FlexTensor::from_data(TensorData::new(vec![0.0f32, 0.0], vec![2]));
        let options = ConvOptions::new([1, 1], [1, 1], [1, 1], 1);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            conv2d_f32(x, weight, Some(bias), &options)
        }));
        assert!(result.is_err(), "expected panic on bias length mismatch");
    }

    #[test]
    fn test_conv2d_depthwise_bias_length_mismatch_panics() {
        // Mirror of the small-channel bias check for the depthwise path.
        // Depthwise dispatch requires groups == channels_in == channels_out, so
        // we construct a 3->3 depthwise conv and deliberately underspecify the
        // bias length to exercise the assert_eq in conv3d_depthwise_impl.
        let x = FlexTensor::from_data(TensorData::new(vec![0.0f32; 48], vec![1, 3, 4, 4]));
        let weight = FlexTensor::from_data(TensorData::new(vec![0.0f32; 27], vec![3, 1, 3, 3]));
        // Bias has only 2 elements but there are 3 depthwise channels.
        let bias = FlexTensor::from_data(TensorData::new(vec![0.0f32, 0.0], vec![2]));
        let options = ConvOptions::new([1, 1], [1, 1], [1, 1], 3); // groups=3 => depthwise
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            conv2d_f32(x, weight, Some(bias), &options)
        }));
        assert!(result.is_err(), "expected panic on bias length mismatch");
    }

    #[test]
    fn test_conv_plane_accumulate_accumulates_into_prefilled() {
        // Pin the documented precondition that `conv_plane_accumulate`
        // accumulates into `out_plane` rather than overwriting it. Both
        // in-tree callers pre-zero before the first call, then call it
        // repeatedly across input channels with the result acting as the
        // running accumulator. A regression that overwrites instead of
        // accumulating would silently drop every input-channel contribution
        // except the last.
        //
        // Test shape: single 2x2 input -> single 2x2 output, 1x1 kernel with
        // weight 1.0. Pre-fill out_plane with [10, 20, 30, 40] so the
        // expected post-call value is x + prefill.
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let w = vec![1.0f32]; // 1x1 kernel, weight = 1
        let mut out_plane = vec![10.0f32, 20.0, 30.0, 40.0];

        // 1x1 kernel, no padding: the only kernel position contributes to
        // every output row and column (indices [0, out_h) and [0, out_w)).
        let oh_ranges = [(0usize, 2usize)]; // per kh: (out_start, out_end)
        let ow_ranges = [(0usize, 2usize)]; // per kw: (out_start, out_end)

        super::conv_plane_accumulate::<f32>(
            &mut out_plane,
            &x,
            &w,
            /* kernel_h */ 1,
            /* kernel_w */ 1,
            /* in_w */ 2,
            /* out_w */ 2,
            /* stride_h */ 1,
            /* stride_w */ 1,
            /* pad_h */ 0,
            /* pad_w */ 0,
            /* dilation_h */ 1,
            /* dilation_w */ 1,
            &oh_ranges,
            &ow_ranges,
        );

        // Expected: prefill + (x * 1.0) element-wise.
        assert_eq!(out_plane, vec![11.0f32, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_conv1d_small_channel_3in() {
        // conv1d -> conv3d expansion with groups=1 and 3 input channels.
        let batch = 2;
        let channels_in = 3;
        let channels_out = 5;
        let in_w = 24;
        let kw = 5;
        let stride = 1;
        let pad = 2;
        let out_w = in_w; // same padding

        let x_data = seeded_vec_f32(batch * channels_in * in_w, 400);
        let w_data = seeded_vec_f32(channels_out * channels_in * kw, 500);
        let x = FlexTensor::from_data(TensorData::new(
            x_data.clone(),
            vec![batch, channels_in, in_w],
        ));
        let weight = FlexTensor::from_data(TensorData::new(
            w_data.clone(),
            vec![channels_out, channels_in, kw],
        ));
        let options = ConvOptions::new([stride], [pad], [1], 1);
        let result = conv1d_f32(x, weight, None, &options);
        assert_eq!(
            result.layout().shape().to_vec(),
            vec![batch, channels_out, out_w]
        );
        let out: Vec<f32> = result.into_data().to_vec().unwrap();

        for b in 0..batch {
            for co in 0..channels_out {
                for o in 0..out_w {
                    let mut expected = 0.0f32;
                    for ci in 0..channels_in {
                        for k in 0..kw {
                            let i = o as isize + k as isize - pad as isize;
                            if i >= 0 && i < in_w as isize {
                                let x_idx = (b * channels_in + ci) * in_w + i as usize;
                                let w_idx = (co * channels_in + ci) * kw + k;
                                expected += x_data[x_idx] * w_data[w_idx];
                            }
                        }
                    }
                    let actual = out[(b * channels_out + co) * out_w + o];
                    assert!(
                        (actual - expected).abs() < 1e-4,
                        "mismatch at b={b}, co={co}, o={o}: expected {expected}, got {actual}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_conv2d_small_channel_single_input() {
        // 1 input channel is allowed (<= threshold) but this also matches
        // many existing conv tests that passed through conv3d_impl before
        // this path existed. Cross-check against the naive reference.
        check_small_channel_conv2d_f32(1, 1, 4, 8, 8, 3, 3, [1, 1], [1, 1], [1, 1], false);
    }

    #[test]
    fn test_conv2d_small_channel_threshold_exact() {
        // Verify the thresholds are honored:
        // - channels_in in [1, 4] AND channels_out in [1, 16] triggers the
        //   small-channel path
        // - channels_in == 5 does not
        // - channels_out == 17 does not
        // - groups != 1 does not
        //
        // conv2d options are expanded to 3D as `[0, pad_h, pad_w]` and
        // `[1, stride_h, stride_w]` etc. - the d-axis is always trivial.

        // c_in == 4, c_out == 16: exactly at the thresholds, should trigger.
        assert!(should_use_small_channel_conv(
            &[1, 4, 1, 8, 8],
            &[16, 4, 1, 3, 3],
            &ConvOptions::new([1, 1, 1], [0, 1, 1], [1, 1, 1], 1),
        ));
        // c_in == 5: excluded.
        assert!(!should_use_small_channel_conv(
            &[1, 5, 1, 8, 8],
            &[2, 5, 1, 3, 3],
            &ConvOptions::new([1, 1, 1], [0, 1, 1], [1, 1, 1], 1),
        ));
        // c_in == 3, c_out == 17: excluded (large output channel count).
        assert!(!should_use_small_channel_conv(
            &[1, 3, 1, 8, 8],
            &[17, 3, 1, 3, 3],
            &ConvOptions::new([1, 1, 1], [0, 1, 1], [1, 1, 1], 1),
        ));
        // c_in == 3, c_out == 64 (ImageNet first layer): excluded.
        assert!(!should_use_small_channel_conv(
            &[1, 3, 1, 224, 224],
            &[64, 3, 1, 7, 7],
            &ConvOptions::new([1, 2, 2], [0, 3, 3], [1, 1, 1], 1),
        ));
        // Non-groups=1 is excluded.
        assert!(!should_use_small_channel_conv(
            &[1, 4, 1, 8, 8],
            &[4, 1, 1, 3, 3],
            &ConvOptions::new([1, 1, 1], [0, 1, 1], [1, 1, 1], 4),
        ));

        // Cross-check c_in == 5 against naive reference (this goes through
        // the generic conv3d_impl path, not the small-channel path).
        check_small_channel_conv2d_f32(1, 5, 4, 8, 8, 3, 3, [1, 1], [1, 1], [1, 1], false);
    }

    // The tests below exercise `conv_plane_accumulate_oh_outer`, the variant
    // selected when the output plane exceeds `CONV_PLANE_OH_OUTER_THRESHOLD`
    // (8192 elements). Every test above this point uses shapes where the plane
    // is <= 256 elements and stays on the kh-outer variant; without these,
    // the oh-outer code path ships uncovered.
    //
    // 96x96 = 9216 > 8192, so one element past the threshold. 97x97 = 9409
    // leaves a little more slack in case someone nudges the threshold.

    #[test]
    fn test_conv2d_depthwise_oh_outer_k3x3() {
        // Depthwise path, oh-outer dispatch, 3x3 kernel, stride 1.
        // Plane = 96 * 96 = 9216 elements.
        check_depthwise_conv2d_f32(1, 2, 96, 96, 3, 3, [1, 1], [1, 1], [1, 1], false);
    }

    #[test]
    fn test_conv2d_depthwise_oh_outer_k3x3_stride2() {
        // Depthwise, oh-outer, stride 2: exercises the `stride_w != 1`
        // inner branch inside the oh-outer variant. Plane = 96*96 = 9216.
        check_depthwise_conv2d_f32(1, 2, 192, 192, 3, 3, [2, 2], [1, 1], [1, 1], false);
    }

    #[test]
    fn test_conv2d_depthwise_oh_outer_k5x1() {
        // Depthwise, oh-outer, 5x1 asymmetric kernel: the exact shape
        // pattern that motivated the loop reorder (Sobel-style separable
        // filter on a large plane). Plane = 97*97 = 9409.
        check_depthwise_conv2d_f32(1, 2, 97, 97, 5, 1, [1, 1], [2, 0], [1, 1], false);
    }

    #[test]
    fn test_conv2d_depthwise_oh_outer_k1x5() {
        // Depthwise, oh-outer, 1x5 asymmetric kernel. Plane = 97*97 = 9409.
        check_depthwise_conv2d_f32(1, 2, 97, 97, 1, 5, [1, 1], [0, 2], [1, 1], false);
    }

    #[test]
    fn test_conv2d_small_channel_oh_outer_k3x3() {
        // Small-channel path, oh-outer dispatch, stride 1.
        // Plane = 96 * 96 = 9216.
        check_small_channel_conv2d_f32(1, 3, 4, 96, 96, 3, 3, [1, 1], [1, 1], [1, 1], false);
    }

    #[test]
    fn test_conv2d_small_channel_oh_outer_k3x3_stride2() {
        // Small-channel, oh-outer, stride 2: exercises the stride != 1
        // inner branch through small-channel dispatch. Plane = 96*96.
        check_small_channel_conv2d_f32(1, 3, 4, 192, 192, 3, 3, [2, 2], [1, 1], [1, 1], false);
    }

    #[test]
    fn test_conv2d_small_channel_oh_outer_k5x1_sobel() {
        // Small-channel, oh-outer, 5x1 asymmetric kernel: the shape from
        // the user-reported Sobel regression on RGB, on a plane large
        // enough to cross the threshold. Plane = 97*97 = 9409.
        check_small_channel_conv2d_f32(1, 3, 3, 97, 97, 5, 1, [1, 1], [2, 0], [1, 1], false);
    }

    #[test]
    fn test_conv2d_small_channel_oh_outer_k1x5_sobel() {
        // Small-channel, oh-outer, 1x5 asymmetric kernel. Plane = 97*97.
        check_small_channel_conv2d_f32(1, 3, 3, 97, 97, 1, 5, [1, 1], [0, 2], [1, 1], false);
    }

    #[test]
    fn test_conv2d_small_channel_oh_outer_with_bias_and_dilation() {
        // Small-channel, oh-outer, with bias and dilation > 1.
        // Plane = 96*96 = 9216.
        check_small_channel_conv2d_f32(1, 3, 8, 100, 100, 3, 3, [1, 1], [2, 2], [2, 2], true);
    }

    #[test]
    fn test_conv2d_depthwise_predicate_triggers() {
        // Directly assert `should_use_depthwise_conv` returns true for
        // representative canonical depthwise shapes, and false for shapes
        // that look depthwise-ish but are not. Without this, a future change
        // that tightens the predicate could silently revert depthwise shapes
        // to the generic path while the correctness tests still pass.
        //
        // conv2d options expand to 3D as `[1, stride_h, stride_w]` etc., with
        // the d-axis always trivial. conv1d expands with `kd == kh == 1` and
        // `in_d == in_h == 1`.

        // Canonical depthwise 3x3, groups == c_in == c_out == 32.
        assert!(should_use_depthwise_conv(
            &[4, 32, 1, 56, 56],
            &[32, 1, 1, 3, 3],
            &ConvOptions::new([1, 1, 1], [0, 1, 1], [1, 1, 1], 32),
        ));
        // Canonical depthwise 7x7, the ConvNeXt shape Thomas reported.
        assert!(should_use_depthwise_conv(
            &[4, 48, 1, 56, 56],
            &[48, 1, 1, 7, 7],
            &ConvOptions::new([1, 1, 1], [0, 3, 3], [1, 1, 1], 48),
        ));
        // Conv1d depthwise (kh == 1 via the conv1d -> conv3d expansion).
        assert!(should_use_depthwise_conv(
            &[8, 64, 1, 1, 1024],
            &[64, 1, 1, 1, 3],
            &ConvOptions::new([1, 1, 1], [0, 0, 1], [1, 1, 1], 64),
        ));
        // Strided + dilated depthwise.
        assert!(should_use_depthwise_conv(
            &[1, 16, 1, 32, 32],
            &[16, 1, 1, 3, 3],
            &ConvOptions::new([1, 2, 2], [0, 1, 1], [1, 2, 2], 16),
        ));

        // Not depthwise: groups == 1.
        assert!(!should_use_depthwise_conv(
            &[1, 8, 1, 16, 16],
            &[16, 8, 1, 3, 3],
            &ConvOptions::new([1, 1, 1], [0, 1, 1], [1, 1, 1], 1),
        ));
        // Not depthwise: channels_per_group > 1 (grouped but not depthwise).
        assert!(!should_use_depthwise_conv(
            &[1, 8, 1, 16, 16],
            &[8, 4, 1, 3, 3],
            &ConvOptions::new([1, 1, 1], [0, 1, 1], [1, 1, 1], 2),
        ));
        // Not depthwise: groups == c_in but c_out != c_in (depth multiplier
        // > 1; the canonical depthwise path is restricted to multiplier 1).
        assert!(!should_use_depthwise_conv(
            &[1, 8, 1, 16, 16],
            &[16, 1, 1, 3, 3],
            &ConvOptions::new([1, 1, 1], [0, 1, 1], [1, 1, 1], 8),
        ));
        // Not depthwise: pure 3D with kd > 1 (the `d`-axis restriction).
        assert!(!should_use_depthwise_conv(
            &[1, 8, 4, 16, 16],
            &[8, 1, 3, 3, 3],
            &ConvOptions::new([1, 1, 1], [1, 1, 1], [1, 1, 1], 8),
        ));
    }

    #[test]
    fn test_valid_out_range_basics() {
        // Sanity checks for the analytic range helper.
        // 3x3, no pad, stride 1: every k position produces the full range.
        let (s, e) = valid_out_range(0, 1, 0, 1, 5, 3);
        assert_eq!((s, e), (0, 3));
        let (s, e) = valid_out_range(2, 1, 0, 1, 5, 3);
        assert_eq!((s, e), (0, 3));
        // 3x3, pad 1, stride 1: kernel position 0 needs o*1 + 0 >= 1 -> o >= 1.
        let (s, e) = valid_out_range(0, 1, 1, 1, 5, 5);
        assert_eq!((s, e), (1, 5));
        // kernel position 2 needs o*1 + 2 - 1 < 5 -> o < 4.
        let (s, e) = valid_out_range(2, 1, 1, 1, 5, 5);
        assert_eq!((s, e), (0, 4));
        // stride 2: o*2 + 0 in [0, in). With in=5, out=3: o in [0, 3).
        let (s, e) = valid_out_range(0, 1, 0, 2, 5, 3);
        assert_eq!((s, e), (0, 3));
        // dilation 2, pad 2, stride 1: kernel position 0 iw = o*1 + 0 - 2 -> o >= 2.
        let (s, e) = valid_out_range(0, 2, 2, 1, 5, 5);
        assert_eq!((s, e), (2, 5));
    }
}
