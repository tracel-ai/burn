//! Pooling operations using a unified 3D implementation.
//!
//! All pooling (1D, 2D, 3D) uses a unified 3D implementation:
//! - pool1d: adds two size-1 dimensions, calls pool3d, squeezes output
//! - pool2d: adds one size-1 dimension, calls pool3d, squeezes output
//! - pool3d: native implementation
//!
//! Supported dtypes: f32, f64, f16 (native), bf16 (via f32 conversion)

use alloc::vec;
use alloc::vec::Vec;
use burn_backend::{DType, Element};
use burn_std::{Bytes, Shape, bf16, f16};

use crate::{FlexTensor, Layout};

// ============================================================================
// Macros for dtype wrappers
// ============================================================================

/// Generates max_pool3d_with_indices typed dispatchers.
macro_rules! max_pool3d_with_indices_typed {
    ($fn_name:ident, $T:ty, $dtype:expr, $neg_inf:expr) => {
        pub fn $fn_name(
            x: FlexTensor,
            kernel_size: [usize; 3],
            stride: [usize; 3],
            padding: [usize; 3],
            dilation: [usize; 3],
            ceil_mode: bool,
        ) -> (FlexTensor, FlexTensor) {
            max_pool3d_with_indices_impl::<$T>(
                x,
                kernel_size,
                stride,
                padding,
                dilation,
                ceil_mode,
                $dtype,
                $neg_inf,
            )
        }
    };
}

/// Generates avg_pool3d typed dispatchers.
macro_rules! avg_pool3d_typed {
    ($fn_name:ident, $T:ty, $dtype:expr, $zero:expr, $add_fn:expr, $div_fn:expr) => {
        pub fn $fn_name(
            x: FlexTensor,
            kernel_size: [usize; 3],
            stride: [usize; 3],
            padding: [usize; 3],
            count_include_pad: bool,
            ceil_mode: bool,
        ) -> FlexTensor {
            avg_pool3d_impl::<$T>(
                x,
                kernel_size,
                stride,
                padding,
                count_include_pad,
                ceil_mode,
                $dtype,
                $zero,
                $add_fn,
                $div_fn,
            )
        }
    };
}

/// Generates adaptive_avg_pool3d typed dispatchers.
macro_rules! adaptive_avg_pool3d_typed {
    ($fn_name:ident, $T:ty, $dtype:expr, $zero:expr, $add_fn:expr, $div_fn:expr) => {
        pub fn $fn_name(x: FlexTensor, output_size: [usize; 3]) -> FlexTensor {
            adaptive_avg_pool3d_impl::<$T>(x, output_size, $dtype, $zero, $add_fn, $div_fn)
        }
    };
}

/// Generates max_pool3d_backward typed dispatchers.
macro_rules! max_pool3d_backward_typed {
    ($fn_name:ident, $T:ty, $dtype:expr, $zero:expr, $add_fn:expr) => {
        pub fn $fn_name(x: FlexTensor, grad: FlexTensor, indices: FlexTensor) -> FlexTensor {
            max_pool3d_backward_impl::<$T>(x, grad, indices, $dtype, $zero, $add_fn)
        }
    };
}

/// Generates avg_pool3d_backward typed dispatchers.
macro_rules! avg_pool3d_backward_typed {
    ($fn_name:ident, $T:ty, $dtype:expr, $zero:expr, $add_fn:expr, $div_fn:expr) => {
        pub fn $fn_name(
            x: FlexTensor,
            grad: FlexTensor,
            kernel_size: [usize; 3],
            stride: [usize; 3],
            padding: [usize; 3],
            count_include_pad: bool,
        ) -> FlexTensor {
            avg_pool3d_backward_impl::<$T>(
                x,
                grad,
                kernel_size,
                stride,
                padding,
                count_include_pad,
                $dtype,
                $zero,
                $add_fn,
                $div_fn,
            )
        }
    };
}

/// Generates adaptive_avg_pool3d_backward typed dispatchers.
macro_rules! adaptive_avg_pool3d_backward_typed {
    ($fn_name:ident, $T:ty, $dtype:expr, $zero:expr, $add_fn:expr, $div_fn:expr) => {
        pub fn $fn_name(x: FlexTensor, grad: FlexTensor) -> FlexTensor {
            adaptive_avg_pool3d_backward_impl::<$T>(x, grad, $dtype, $zero, $add_fn, $div_fn)
        }
    };
}

// ============================================================================
// Output size calculation
// ============================================================================

/// Calculate pooling output size for a single dimension.
fn pool_output_size(
    input: usize,
    kernel: usize,
    padding: usize,
    stride: usize,
    dilation: usize,
    ceil_mode: bool,
) -> usize {
    assert!(kernel > 0, "pool: kernel size must be > 0");
    assert!(stride > 0, "pool: stride must be > 0");
    let effective_kernel = dilation * (kernel - 1) + 1;
    let padded = input + 2 * padding;
    if padded < effective_kernel {
        return if ceil_mode { 1 } else { 0 };
    }
    let numerator = padded - effective_kernel;
    if ceil_mode {
        numerator.div_ceil(stride) + 1
    } else {
        numerator / stride + 1
    }
}

// ============================================================================
// Max Pool 3D - core implementation
// ============================================================================

max_pool3d_with_indices_typed!(
    max_pool3d_with_indices_f32,
    f32,
    DType::F32,
    f32::NEG_INFINITY
);
max_pool3d_with_indices_typed!(
    max_pool3d_with_indices_f64,
    f64,
    DType::F64,
    f64::NEG_INFINITY
);
max_pool3d_with_indices_typed!(
    max_pool3d_with_indices_f16,
    f16,
    DType::F16,
    f16::NEG_INFINITY
);

pub fn max_pool3d_with_indices_bf16(
    x: FlexTensor,
    kernel_size: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
    dilation: [usize; 3],
    ceil_mode: bool,
) -> (FlexTensor, FlexTensor) {
    let x_f32 = convert_bf16_to_f32(&x);
    let (output_f32, indices) =
        max_pool3d_with_indices_f32(x_f32, kernel_size, stride, padding, dilation, ceil_mode);
    (convert_f32_to_bf16(&output_f32), indices)
}

/// Generic 3D max pooling with indices implementation.
#[allow(clippy::too_many_arguments)]
fn max_pool3d_with_indices_impl<T>(
    x: FlexTensor,
    kernel_size: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
    dilation: [usize; 3],
    ceil_mode: bool,
    dtype: DType,
    neg_inf: T,
) -> (FlexTensor, FlexTensor)
where
    T: bytemuck::Pod + Copy + PartialOrd + Send + Sync + Element,
{
    let x = x.to_contiguous();
    let x_shape = x.layout().shape();

    let batch_size = x_shape[0];
    let channels = x_shape[1];
    let in_d = x_shape[2];
    let in_h = x_shape[3];
    let in_w = x_shape[4];

    let [kernel_d, kernel_h, kernel_w] = kernel_size;
    let [stride_d, stride_h, stride_w] = stride;
    let [pad_d, pad_h, pad_w] = padding;
    let [dilation_d, dilation_h, dilation_w] = dilation;

    let out_d = pool_output_size(in_d, kernel_d, pad_d, stride_d, dilation_d, ceil_mode);
    let out_h = pool_output_size(in_h, kernel_h, pad_h, stride_h, dilation_h, ceil_mode);
    let out_w = pool_output_size(in_w, kernel_w, pad_w, stride_w, dilation_w, ceil_mode);

    let spatial_out = out_d * out_h * out_w;
    let x_data: &[T] = x.storage();

    let (output, indices) = {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;

            let mut output = vec![neg_inf; batch_size * channels * spatial_out];
            let mut indices = vec![-1i64; batch_size * channels * spatial_out];
            let out_ptr = crate::ops::SendMutPtr::new(output.as_mut_ptr());
            let idx_ptr = crate::ops::SendMutPtr::new(indices.as_mut_ptr());

            // Flatten batch*channels into a single par_iter so rayon chooses
            // the right granularity instead of creating one task per channel.
            let bc_total = batch_size * channels;
            (0..bc_total).into_par_iter().for_each(|bc| {
                let b = bc / channels;
                let c = bc % channels;
                let x_offset = b * channels * in_d * in_h * in_w + c * in_d * in_h * in_w;
                let out_offset = bc * spatial_out;

                for od in 0..out_d {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let out_idx = out_offset + od * out_h * out_w + oh * out_w + ow;
                            let mut max_val = neg_inf;
                            let mut max_idx: i64 = -1;

                            for kd in 0..kernel_d {
                                let id =
                                    (od * stride_d + kd * dilation_d) as isize - pad_d as isize;
                                if id < 0 || id >= in_d as isize {
                                    continue;
                                }
                                let id = id as usize;

                                for kh in 0..kernel_h {
                                    let ih =
                                        (oh * stride_h + kh * dilation_h) as isize - pad_h as isize;
                                    if ih < 0 || ih >= in_h as isize {
                                        continue;
                                    }
                                    let ih = ih as usize;

                                    for kw in 0..kernel_w {
                                        let iw = (ow * stride_w + kw * dilation_w) as isize
                                            - pad_w as isize;
                                        if iw < 0 || iw >= in_w as isize {
                                            continue;
                                        }
                                        let iw = iw as usize;

                                        let x_idx = x_offset + id * in_h * in_w + ih * in_w + iw;
                                        let val = x_data[x_idx];

                                        if max_idx < 0 || val > max_val {
                                            max_val = val;
                                            max_idx = (id * in_h * in_w + ih * in_w + iw) as i64;
                                        }
                                    }
                                }
                            }

                            unsafe {
                                out_ptr.write(out_idx, max_val);
                                idx_ptr.write(out_idx, max_idx);
                            }
                        }
                    }
                }
            });
            (output, indices)
        }
        #[cfg(not(feature = "rayon"))]
        {
            let mut output = vec![neg_inf; batch_size * channels * spatial_out];
            let mut indices = vec![-1i64; batch_size * channels * spatial_out];

            for b in 0..batch_size {
                for c in 0..channels {
                    let x_offset = b * channels * in_d * in_h * in_w + c * in_d * in_h * in_w;
                    let out_offset = b * channels * spatial_out + c * spatial_out;

                    for od in 0..out_d {
                        for oh in 0..out_h {
                            for ow in 0..out_w {
                                let out_idx = out_offset + od * out_h * out_w + oh * out_w + ow;
                                let mut max_val = neg_inf;
                                let mut max_idx: i64 = -1;

                                for kd in 0..kernel_d {
                                    let id =
                                        (od * stride_d + kd * dilation_d) as isize - pad_d as isize;
                                    if id < 0 || id >= in_d as isize {
                                        continue;
                                    }
                                    let id = id as usize;

                                    for kh in 0..kernel_h {
                                        let ih = (oh * stride_h + kh * dilation_h) as isize
                                            - pad_h as isize;
                                        if ih < 0 || ih >= in_h as isize {
                                            continue;
                                        }
                                        let ih = ih as usize;

                                        for kw in 0..kernel_w {
                                            let iw = (ow * stride_w + kw * dilation_w) as isize
                                                - pad_w as isize;
                                            if iw < 0 || iw >= in_w as isize {
                                                continue;
                                            }
                                            let iw = iw as usize;

                                            let x_idx =
                                                x_offset + id * in_h * in_w + ih * in_w + iw;
                                            let val = x_data[x_idx];

                                            if max_idx < 0 || val > max_val {
                                                max_val = val;
                                                max_idx =
                                                    (id * in_h * in_w + ih * in_w + iw) as i64;
                                            }
                                        }
                                    }
                                }

                                output[out_idx] = max_val;
                                indices[out_idx] = max_idx;
                            }
                        }
                    }
                }
            }
            (output, indices)
        }
    };

    let out_shape = Shape::from(vec![batch_size, channels, out_d, out_h, out_w]);
    let output_tensor = FlexTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(out_shape.clone()),
        dtype,
    );
    let indices_tensor = FlexTensor::new(
        Bytes::from_elems(indices),
        Layout::contiguous(out_shape),
        DType::I64,
    );

    (output_tensor, indices_tensor)
}

/// 3D max pooling (without returning indices) for f32.
pub fn max_pool3d_f32(
    x: FlexTensor,
    kernel_size: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
    dilation: [usize; 3],
    ceil_mode: bool,
) -> FlexTensor {
    max_pool3d_with_indices_f32(x, kernel_size, stride, padding, dilation, ceil_mode).0
}

/// 3D max pooling (without returning indices) for f64.
pub fn max_pool3d_f64(
    x: FlexTensor,
    kernel_size: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
    dilation: [usize; 3],
    ceil_mode: bool,
) -> FlexTensor {
    max_pool3d_with_indices_f64(x, kernel_size, stride, padding, dilation, ceil_mode).0
}

/// 3D max pooling (without returning indices) for f16.
pub fn max_pool3d_f16(
    x: FlexTensor,
    kernel_size: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
    dilation: [usize; 3],
    ceil_mode: bool,
) -> FlexTensor {
    max_pool3d_with_indices_f16(x, kernel_size, stride, padding, dilation, ceil_mode).0
}

/// 3D max pooling (without returning indices) for bf16.
pub fn max_pool3d_bf16(
    x: FlexTensor,
    kernel_size: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
    dilation: [usize; 3],
    ceil_mode: bool,
) -> FlexTensor {
    max_pool3d_with_indices_bf16(x, kernel_size, stride, padding, dilation, ceil_mode).0
}

// ============================================================================
// Max Pool 2D - delegates to 3D
// ============================================================================

/// 2D max pooling with indices for f32.
pub fn max_pool2d_with_indices_f32(
    x: FlexTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
) -> (FlexTensor, FlexTensor) {
    let x_3d = expand_2d_to_3d(&x);
    let (output, indices) = max_pool3d_with_indices_f32(
        x_3d,
        [1, kernel_size[0], kernel_size[1]],
        [1, stride[0], stride[1]],
        [0, padding[0], padding[1]],
        [1, dilation[0], dilation[1]],
        ceil_mode,
    );
    (squeeze_3d_to_2d(output), squeeze_3d_to_2d(indices))
}

/// 2D max pooling with indices for f64.
pub fn max_pool2d_with_indices_f64(
    x: FlexTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
) -> (FlexTensor, FlexTensor) {
    let x_3d = expand_2d_to_3d(&x);
    let (output, indices) = max_pool3d_with_indices_f64(
        x_3d,
        [1, kernel_size[0], kernel_size[1]],
        [1, stride[0], stride[1]],
        [0, padding[0], padding[1]],
        [1, dilation[0], dilation[1]],
        ceil_mode,
    );
    (squeeze_3d_to_2d(output), squeeze_3d_to_2d(indices))
}

/// 2D max pooling with indices for f16.
pub fn max_pool2d_with_indices_f16(
    x: FlexTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
) -> (FlexTensor, FlexTensor) {
    let x_3d = expand_2d_to_3d(&x);
    let (output, indices) = max_pool3d_with_indices_f16(
        x_3d,
        [1, kernel_size[0], kernel_size[1]],
        [1, stride[0], stride[1]],
        [0, padding[0], padding[1]],
        [1, dilation[0], dilation[1]],
        ceil_mode,
    );
    (squeeze_3d_to_2d(output), squeeze_3d_to_2d(indices))
}

/// 2D max pooling with indices for bf16.
pub fn max_pool2d_with_indices_bf16(
    x: FlexTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
) -> (FlexTensor, FlexTensor) {
    let x_3d = expand_2d_to_3d(&x);
    let (output, indices) = max_pool3d_with_indices_bf16(
        x_3d,
        [1, kernel_size[0], kernel_size[1]],
        [1, stride[0], stride[1]],
        [0, padding[0], padding[1]],
        [1, dilation[0], dilation[1]],
        ceil_mode,
    );
    (squeeze_3d_to_2d(output), squeeze_3d_to_2d(indices))
}

/// 2D max pooling (without indices) for f32.
pub fn max_pool2d_f32(
    x: FlexTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
) -> FlexTensor {
    max_pool2d_with_indices_f32(x, kernel_size, stride, padding, dilation, ceil_mode).0
}

/// 2D max pooling (without indices) for f64.
pub fn max_pool2d_f64(
    x: FlexTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
) -> FlexTensor {
    max_pool2d_with_indices_f64(x, kernel_size, stride, padding, dilation, ceil_mode).0
}

/// 2D max pooling (without indices) for f16.
pub fn max_pool2d_f16(
    x: FlexTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
) -> FlexTensor {
    max_pool2d_with_indices_f16(x, kernel_size, stride, padding, dilation, ceil_mode).0
}

/// 2D max pooling (without indices) for bf16.
pub fn max_pool2d_bf16(
    x: FlexTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
) -> FlexTensor {
    max_pool2d_with_indices_bf16(x, kernel_size, stride, padding, dilation, ceil_mode).0
}

// ============================================================================
// Max Pool 1D - delegates to 3D
// ============================================================================

/// 1D max pooling with indices for f32.
pub fn max_pool1d_with_indices_f32(
    x: FlexTensor,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    ceil_mode: bool,
) -> (FlexTensor, FlexTensor) {
    let x_3d = expand_1d_to_3d(&x);
    let (output, indices) = max_pool3d_with_indices_f32(
        x_3d,
        [1, 1, kernel_size],
        [1, 1, stride],
        [0, 0, padding],
        [1, 1, dilation],
        ceil_mode,
    );
    (squeeze_3d_to_1d(output), squeeze_3d_to_1d(indices))
}

/// 1D max pooling (without indices) for f32.
pub fn max_pool1d_f32(
    x: FlexTensor,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    ceil_mode: bool,
) -> FlexTensor {
    max_pool1d_with_indices_f32(x, kernel_size, stride, padding, dilation, ceil_mode).0
}

// ============================================================================
// Avg Pool 3D - core implementation
// ============================================================================

avg_pool3d_typed!(
    avg_pool3d_f32,
    f32,
    DType::F32,
    0.0f32,
    |a, b| a + b,
    |sum, count| sum / count as f32
);
avg_pool3d_typed!(
    avg_pool3d_f64,
    f64,
    DType::F64,
    0.0f64,
    |a, b| a + b,
    |sum, count| sum / count as f64
);
avg_pool3d_typed!(
    avg_pool3d_f16,
    f16,
    DType::F16,
    f16::from_f32(0.0),
    |a: f16, b: f16| f16::from_f32(a.to_f32() + b.to_f32()),
    |sum: f16, count| f16::from_f32(sum.to_f32() / count as f32)
);

pub fn avg_pool3d_bf16(
    x: FlexTensor,
    kernel_size: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
    count_include_pad: bool,
    ceil_mode: bool,
) -> FlexTensor {
    let x_f32 = convert_bf16_to_f32(&x);
    let result_f32 = avg_pool3d_f32(
        x_f32,
        kernel_size,
        stride,
        padding,
        count_include_pad,
        ceil_mode,
    );
    convert_f32_to_bf16(&result_f32)
}

/// Generic 3D average pooling implementation.
#[allow(clippy::too_many_arguments)]
fn avg_pool3d_impl<T>(
    x: FlexTensor,
    kernel_size: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
    count_include_pad: bool,
    ceil_mode: bool,
    dtype: DType,
    zero: T,
    add_fn: fn(T, T) -> T,
    div_fn: fn(T, usize) -> T,
) -> FlexTensor
where
    T: bytemuck::Pod + Copy + Send + Sync + Element,
{
    let x = x.to_contiguous();
    let x_shape = x.layout().shape();

    let batch_size = x_shape[0];
    let channels = x_shape[1];
    let in_d = x_shape[2];
    let in_h = x_shape[3];
    let in_w = x_shape[4];

    let [kernel_d, kernel_h, kernel_w] = kernel_size;
    let [stride_d, stride_h, stride_w] = stride;
    let [pad_d, pad_h, pad_w] = padding;

    // Avg pool doesn't use dilation in typical implementations
    let dilation_d = 1;
    let dilation_h = 1;
    let dilation_w = 1;

    let out_d = pool_output_size(in_d, kernel_d, pad_d, stride_d, dilation_d, ceil_mode);
    let out_h = pool_output_size(in_h, kernel_h, pad_h, stride_h, dilation_h, ceil_mode);
    let out_w = pool_output_size(in_w, kernel_w, pad_w, stride_w, dilation_w, ceil_mode);

    let spatial_out = out_d * out_h * out_w;
    let x_data: &[T] = x.storage();
    let _kernel_volume = kernel_d * kernel_h * kernel_w;

    let output = {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;

            let mut output = vec![zero; batch_size * channels * spatial_out];
            let out_ptr = crate::ops::SendMutPtr::new(output.as_mut_ptr());

            let bc_total = batch_size * channels;
            (0..bc_total).into_par_iter().for_each(|bc| {
                let b = bc / channels;
                let c = bc % channels;
                let x_offset = b * channels * in_d * in_h * in_w + c * in_d * in_h * in_w;
                let out_offset = bc * spatial_out;

                for od in 0..out_d {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let out_idx = out_offset + od * out_h * out_w + oh * out_w + ow;
                            let mut sum = zero;
                            let mut count = 0usize;
                            let mut pad_count = 0usize;

                            for kd in 0..kernel_d {
                                let id = (od * stride_d + kd) as isize - pad_d as isize;
                                let id_in_bounds =
                                    id >= -(pad_d as isize) && id < (in_d + pad_d) as isize;
                                if !id_in_bounds {
                                    continue;
                                }
                                let id_valid = id >= 0 && id < in_d as isize;

                                for kh in 0..kernel_h {
                                    let ih = (oh * stride_h + kh) as isize - pad_h as isize;
                                    let ih_in_bounds =
                                        ih >= -(pad_h as isize) && ih < (in_h + pad_h) as isize;
                                    if !ih_in_bounds {
                                        continue;
                                    }
                                    let ih_valid = ih >= 0 && ih < in_h as isize;

                                    for kw in 0..kernel_w {
                                        let iw = (ow * stride_w + kw) as isize - pad_w as isize;
                                        let iw_in_bounds =
                                            iw >= -(pad_w as isize) && iw < (in_w + pad_w) as isize;
                                        if !iw_in_bounds {
                                            continue;
                                        }

                                        pad_count += 1;

                                        let iw_valid = iw >= 0 && iw < in_w as isize;
                                        if !id_valid || !ih_valid || !iw_valid {
                                            continue;
                                        }

                                        let id = id as usize;
                                        let ih = ih as usize;
                                        let iw = iw as usize;
                                        let x_idx = x_offset + id * in_h * in_w + ih * in_w + iw;
                                        sum = add_fn(sum, x_data[x_idx]);
                                        count += 1;
                                    }
                                }
                            }

                            let divisor = if count_include_pad {
                                pad_count.max(1)
                            } else {
                                count.max(1)
                            };

                            unsafe {
                                out_ptr.write(out_idx, div_fn(sum, divisor));
                            }
                        }
                    }
                }
            });
            output
        }
        #[cfg(not(feature = "rayon"))]
        {
            let mut output = vec![zero; batch_size * channels * spatial_out];

            for b in 0..batch_size {
                for c in 0..channels {
                    let x_offset = b * channels * in_d * in_h * in_w + c * in_d * in_h * in_w;
                    let out_offset = b * channels * spatial_out + c * spatial_out;

                    for od in 0..out_d {
                        for oh in 0..out_h {
                            for ow in 0..out_w {
                                let out_idx = out_offset + od * out_h * out_w + oh * out_w + ow;
                                let mut sum = zero;
                                let mut count = 0usize;

                                // Track count for count_include_pad (positions within padded bounds)
                                let mut pad_count = 0usize;

                                for kd in 0..kernel_d {
                                    let id = (od * stride_d + kd) as isize - pad_d as isize;
                                    // Check if within padded bounds (not ceil_mode extension)
                                    let id_in_bounds =
                                        id >= -(pad_d as isize) && id < (in_d + pad_d) as isize;
                                    if !id_in_bounds {
                                        continue; // ceil_mode extension - skip entirely
                                    }
                                    let id_valid = id >= 0 && id < in_d as isize;

                                    for kh in 0..kernel_h {
                                        let ih = (oh * stride_h + kh) as isize - pad_h as isize;
                                        let ih_in_bounds =
                                            ih >= -(pad_h as isize) && ih < (in_h + pad_h) as isize;
                                        if !ih_in_bounds {
                                            continue;
                                        }
                                        let ih_valid = ih >= 0 && ih < in_h as isize;

                                        for kw in 0..kernel_w {
                                            let iw = (ow * stride_w + kw) as isize - pad_w as isize;
                                            let iw_in_bounds = iw >= -(pad_w as isize)
                                                && iw < (in_w + pad_w) as isize;
                                            if !iw_in_bounds {
                                                continue;
                                            }

                                            // Position is within padded bounds
                                            pad_count += 1;

                                            let iw_valid = iw >= 0 && iw < in_w as isize;
                                            if !id_valid || !ih_valid || !iw_valid {
                                                continue; // In padding zone - count but don't add
                                            }

                                            let id = id as usize;
                                            let ih = ih as usize;
                                            let iw = iw as usize;
                                            let x_idx =
                                                x_offset + id * in_h * in_w + ih * in_w + iw;
                                            sum = add_fn(sum, x_data[x_idx]);
                                            count += 1;
                                        }
                                    }
                                }

                                let divisor = if count_include_pad {
                                    pad_count.max(1) // Positions within padded bounds
                                } else {
                                    count.max(1) // Only actual valid positions
                                };

                                output[out_idx] = div_fn(sum, divisor);
                            }
                        }
                    }
                }
            }
            output
        }
    };

    let out_shape = Shape::from(vec![batch_size, channels, out_d, out_h, out_w]);
    FlexTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(out_shape),
        dtype,
    )
}

// ============================================================================
// Avg Pool 2D - delegates to 3D
// ============================================================================

/// 2D average pooling for f32.
pub fn avg_pool2d_f32(
    x: FlexTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
    ceil_mode: bool,
) -> FlexTensor {
    let x_3d = expand_2d_to_3d(&x);
    let result = avg_pool3d_f32(
        x_3d,
        [1, kernel_size[0], kernel_size[1]],
        [1, stride[0], stride[1]],
        [0, padding[0], padding[1]],
        count_include_pad,
        ceil_mode,
    );
    squeeze_3d_to_2d(result)
}

/// 2D average pooling for f64.
pub fn avg_pool2d_f64(
    x: FlexTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
    ceil_mode: bool,
) -> FlexTensor {
    let x_3d = expand_2d_to_3d(&x);
    let result = avg_pool3d_f64(
        x_3d,
        [1, kernel_size[0], kernel_size[1]],
        [1, stride[0], stride[1]],
        [0, padding[0], padding[1]],
        count_include_pad,
        ceil_mode,
    );
    squeeze_3d_to_2d(result)
}

/// 2D average pooling for f16.
pub fn avg_pool2d_f16(
    x: FlexTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
    ceil_mode: bool,
) -> FlexTensor {
    let x_3d = expand_2d_to_3d(&x);
    let result = avg_pool3d_f16(
        x_3d,
        [1, kernel_size[0], kernel_size[1]],
        [1, stride[0], stride[1]],
        [0, padding[0], padding[1]],
        count_include_pad,
        ceil_mode,
    );
    squeeze_3d_to_2d(result)
}

/// 2D average pooling for bf16.
pub fn avg_pool2d_bf16(
    x: FlexTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
    ceil_mode: bool,
) -> FlexTensor {
    let x_3d = expand_2d_to_3d(&x);
    let result = avg_pool3d_bf16(
        x_3d,
        [1, kernel_size[0], kernel_size[1]],
        [1, stride[0], stride[1]],
        [0, padding[0], padding[1]],
        count_include_pad,
        ceil_mode,
    );
    squeeze_3d_to_2d(result)
}

// ============================================================================
// Avg Pool 1D - delegates to 3D
// ============================================================================

/// 1D average pooling for f32.
pub fn avg_pool1d_f32(
    x: FlexTensor,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    count_include_pad: bool,
    ceil_mode: bool,
) -> FlexTensor {
    let x_3d = expand_1d_to_3d(&x);
    let result = avg_pool3d_f32(
        x_3d,
        [1, 1, kernel_size],
        [1, 1, stride],
        [0, 0, padding],
        count_include_pad,
        ceil_mode,
    );
    squeeze_3d_to_1d(result)
}

// ============================================================================
// Adaptive Avg Pool 3D - core implementation
// ============================================================================

adaptive_avg_pool3d_typed!(
    adaptive_avg_pool3d_f32,
    f32,
    DType::F32,
    0.0f32,
    |a, b| a + b,
    |sum, count| sum / count as f32
);
adaptive_avg_pool3d_typed!(
    adaptive_avg_pool3d_f64,
    f64,
    DType::F64,
    0.0f64,
    |a, b| a + b,
    |sum, count| sum / count as f64
);
adaptive_avg_pool3d_typed!(
    adaptive_avg_pool3d_f16,
    f16,
    DType::F16,
    f16::from_f32(0.0),
    |a: f16, b: f16| f16::from_f32(a.to_f32() + b.to_f32()),
    |sum: f16, count| f16::from_f32(sum.to_f32() / count as f32)
);

pub fn adaptive_avg_pool3d_bf16(x: FlexTensor, output_size: [usize; 3]) -> FlexTensor {
    let x_f32 = convert_bf16_to_f32(&x);
    let result_f32 = adaptive_avg_pool3d_f32(x_f32, output_size);
    convert_f32_to_bf16(&result_f32)
}

/// Generic 3D adaptive average pooling implementation.
fn adaptive_avg_pool3d_impl<T>(
    x: FlexTensor,
    output_size: [usize; 3],
    dtype: DType,
    zero: T,
    add_fn: fn(T, T) -> T,
    div_fn: fn(T, usize) -> T,
) -> FlexTensor
where
    T: bytemuck::Pod + Copy + Send + Sync + Element,
{
    let x = x.to_contiguous();
    let x_shape = x.layout().shape();

    let batch_size = x_shape[0];
    let channels = x_shape[1];
    let in_d = x_shape[2];
    let in_h = x_shape[3];
    let in_w = x_shape[4];

    let [out_d, out_h, out_w] = output_size;
    let spatial_out = out_d * out_h * out_w;
    let x_data: &[T] = x.storage();

    let output = {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;

            let mut output = vec![zero; batch_size * channels * spatial_out];
            let out_ptr = crate::ops::SendMutPtr::new(output.as_mut_ptr());

            (0..batch_size).into_par_iter().for_each(|b| {
                (0..channels).into_par_iter().for_each(|c| {
                    let x_offset = b * channels * in_d * in_h * in_w + c * in_d * in_h * in_w;
                    let out_offset = b * channels * spatial_out + c * spatial_out;

                    for od in 0..out_d {
                        // Compute input range for this output position
                        // start = floor(out * in / out_size), end = ceil((out+1) * in / out_size)
                        let d_start = (od * in_d) / out_d;
                        let d_end = ((od + 1) * in_d).div_ceil(out_d);

                        for oh in 0..out_h {
                            let h_start = (oh * in_h) / out_h;
                            let h_end = ((oh + 1) * in_h).div_ceil(out_h);

                            for ow in 0..out_w {
                                let w_start = (ow * in_w) / out_w;
                                let w_end = ((ow + 1) * in_w).div_ceil(out_w);

                                let out_idx = out_offset + od * out_h * out_w + oh * out_w + ow;
                                let mut sum = zero;
                                let mut count = 0usize;

                                for id in d_start..d_end {
                                    for ih in h_start..h_end {
                                        for iw in w_start..w_end {
                                            let x_idx =
                                                x_offset + id * in_h * in_w + ih * in_w + iw;
                                            sum = add_fn(sum, x_data[x_idx]);
                                            count += 1;
                                        }
                                    }
                                }

                                unsafe {
                                    out_ptr.write(out_idx, div_fn(sum, count.max(1)));
                                }
                            }
                        }
                    }
                });
            });
            output
        }
        #[cfg(not(feature = "rayon"))]
        {
            let mut output = vec![zero; batch_size * channels * spatial_out];

            for b in 0..batch_size {
                for c in 0..channels {
                    let x_offset = b * channels * in_d * in_h * in_w + c * in_d * in_h * in_w;
                    let out_offset = b * channels * spatial_out + c * spatial_out;

                    for od in 0..out_d {
                        let d_start = (od * in_d) / out_d;
                        let d_end = ((od + 1) * in_d).div_ceil(out_d);

                        for oh in 0..out_h {
                            let h_start = (oh * in_h) / out_h;
                            let h_end = ((oh + 1) * in_h).div_ceil(out_h);

                            for ow in 0..out_w {
                                let w_start = (ow * in_w) / out_w;
                                let w_end = ((ow + 1) * in_w).div_ceil(out_w);

                                let out_idx = out_offset + od * out_h * out_w + oh * out_w + ow;
                                let mut sum = zero;
                                let mut count = 0usize;

                                for id in d_start..d_end {
                                    for ih in h_start..h_end {
                                        for iw in w_start..w_end {
                                            let x_idx =
                                                x_offset + id * in_h * in_w + ih * in_w + iw;
                                            sum = add_fn(sum, x_data[x_idx]);
                                            count += 1;
                                        }
                                    }
                                }

                                output[out_idx] = div_fn(sum, count.max(1));
                            }
                        }
                    }
                }
            }
            output
        }
    };

    let out_shape = Shape::from(vec![batch_size, channels, out_d, out_h, out_w]);
    FlexTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(out_shape),
        dtype,
    )
}

// ============================================================================
// Adaptive Avg Pool 2D - delegates to 3D
// ============================================================================

/// 2D adaptive average pooling for f32.
pub fn adaptive_avg_pool2d_f32(x: FlexTensor, output_size: [usize; 2]) -> FlexTensor {
    let x_3d = expand_2d_to_3d(&x);
    let result = adaptive_avg_pool3d_f32(x_3d, [1, output_size[0], output_size[1]]);
    squeeze_3d_to_2d(result)
}

/// 2D adaptive average pooling for f64.
pub fn adaptive_avg_pool2d_f64(x: FlexTensor, output_size: [usize; 2]) -> FlexTensor {
    let x_3d = expand_2d_to_3d(&x);
    let result = adaptive_avg_pool3d_f64(x_3d, [1, output_size[0], output_size[1]]);
    squeeze_3d_to_2d(result)
}

/// 2D adaptive average pooling for f16.
pub fn adaptive_avg_pool2d_f16(x: FlexTensor, output_size: [usize; 2]) -> FlexTensor {
    let x_3d = expand_2d_to_3d(&x);
    let result = adaptive_avg_pool3d_f16(x_3d, [1, output_size[0], output_size[1]]);
    squeeze_3d_to_2d(result)
}

/// 2D adaptive average pooling for bf16.
pub fn adaptive_avg_pool2d_bf16(x: FlexTensor, output_size: [usize; 2]) -> FlexTensor {
    let x_3d = expand_2d_to_3d(&x);
    let result = adaptive_avg_pool3d_bf16(x_3d, [1, output_size[0], output_size[1]]);
    squeeze_3d_to_2d(result)
}

// ============================================================================
// Adaptive Avg Pool 1D - delegates to 3D
// ============================================================================

/// 1D adaptive average pooling for f32.
pub fn adaptive_avg_pool1d_f32(x: FlexTensor, output_size: usize) -> FlexTensor {
    let x_3d = expand_1d_to_3d(&x);
    let result = adaptive_avg_pool3d_f32(x_3d, [1, 1, output_size]);
    squeeze_3d_to_1d(result)
}

// ============================================================================
// Backward passes
// ============================================================================

/// Max pool 2D backward using stored indices.
pub fn max_pool2d_backward_f32(x: FlexTensor, grad: FlexTensor, indices: FlexTensor) -> FlexTensor {
    let x_3d = expand_2d_to_3d(&x);
    let grad_3d = expand_2d_to_3d(&grad);
    let indices_3d = expand_2d_to_3d(&indices);
    let result = max_pool3d_backward_f32(x_3d, grad_3d, indices_3d);
    squeeze_3d_to_2d(result)
}

/// Max pool 2D backward for f64.
pub fn max_pool2d_backward_f64(x: FlexTensor, grad: FlexTensor, indices: FlexTensor) -> FlexTensor {
    let x_3d = expand_2d_to_3d(&x);
    let grad_3d = expand_2d_to_3d(&grad);
    let indices_3d = expand_2d_to_3d(&indices);
    let result = max_pool3d_backward_f64(x_3d, grad_3d, indices_3d);
    squeeze_3d_to_2d(result)
}

/// Max pool 2D backward for f16.
pub fn max_pool2d_backward_f16(x: FlexTensor, grad: FlexTensor, indices: FlexTensor) -> FlexTensor {
    let x_3d = expand_2d_to_3d(&x);
    let grad_3d = expand_2d_to_3d(&grad);
    let indices_3d = expand_2d_to_3d(&indices);
    let result = max_pool3d_backward_f16(x_3d, grad_3d, indices_3d);
    squeeze_3d_to_2d(result)
}

/// Max pool 2D backward for bf16.
pub fn max_pool2d_backward_bf16(
    x: FlexTensor,
    grad: FlexTensor,
    indices: FlexTensor,
) -> FlexTensor {
    let x_f32 = convert_bf16_to_f32(&x);
    let grad_f32 = convert_bf16_to_f32(&grad);
    let result_f32 = max_pool2d_backward_f32(x_f32, grad_f32, indices);
    convert_f32_to_bf16(&result_f32)
}

max_pool3d_backward_typed!(max_pool3d_backward_f32, f32, DType::F32, 0.0f32, |a, b| a
    + b);
max_pool3d_backward_typed!(max_pool3d_backward_f64, f64, DType::F64, 0.0f64, |a, b| a
    + b);
max_pool3d_backward_typed!(
    max_pool3d_backward_f16,
    f16,
    DType::F16,
    f16::from_f32(0.0),
    |a: f16, b: f16| f16::from_f32(a.to_f32() + b.to_f32())
);

/// Generic max pool 3D backward implementation.
fn max_pool3d_backward_impl<T>(
    x: FlexTensor,
    grad: FlexTensor,
    indices: FlexTensor,
    dtype: DType,
    zero: T,
    add_fn: fn(T, T) -> T,
) -> FlexTensor
where
    T: bytemuck::Pod + Copy + Send + Sync + Element,
{
    let x_shape = x.layout().shape();
    let grad = grad.to_contiguous();
    let indices = indices.to_contiguous();

    let batch_size = x_shape[0];
    let channels = x_shape[1];
    let in_d = x_shape[2];
    let in_h = x_shape[3];
    let in_w = x_shape[4];
    let spatial_in = in_d * in_h * in_w;

    let grad_shape = grad.layout().shape();
    let out_d = grad_shape[2];
    let out_h = grad_shape[3];
    let out_w = grad_shape[4];
    let spatial_out = out_d * out_h * out_w;

    let grad_data: &[T] = grad.storage();
    let indices_data: &[i64] = indices.storage();

    // Accumulate gradients back to input positions
    let mut output = vec![zero; batch_size * channels * spatial_in];

    for b in 0..batch_size {
        for c in 0..channels {
            let grad_offset = b * channels * spatial_out + c * spatial_out;
            let out_offset = b * channels * spatial_in + c * spatial_in;

            for i in 0..spatial_out {
                let idx = indices_data[grad_offset + i];
                if idx >= 0 {
                    let input_idx = out_offset + idx as usize;
                    output[input_idx] = add_fn(output[input_idx], grad_data[grad_offset + i]);
                }
            }
        }
    }

    let out_shape = Shape::from(vec![batch_size, channels, in_d, in_h, in_w]);
    FlexTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(out_shape),
        dtype,
    )
}

/// Avg pool 2D backward.
pub fn avg_pool2d_backward_f32(
    x: FlexTensor,
    grad: FlexTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
) -> FlexTensor {
    let x_3d = expand_2d_to_3d(&x);
    let grad_3d = expand_2d_to_3d(&grad);
    let result = avg_pool3d_backward_f32(
        x_3d,
        grad_3d,
        [1, kernel_size[0], kernel_size[1]],
        [1, stride[0], stride[1]],
        [0, padding[0], padding[1]],
        count_include_pad,
    );
    squeeze_3d_to_2d(result)
}

/// Avg pool 2D backward for f64.
pub fn avg_pool2d_backward_f64(
    x: FlexTensor,
    grad: FlexTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
) -> FlexTensor {
    let x_3d = expand_2d_to_3d(&x);
    let grad_3d = expand_2d_to_3d(&grad);
    let result = avg_pool3d_backward_f64(
        x_3d,
        grad_3d,
        [1, kernel_size[0], kernel_size[1]],
        [1, stride[0], stride[1]],
        [0, padding[0], padding[1]],
        count_include_pad,
    );
    squeeze_3d_to_2d(result)
}

/// Avg pool 2D backward for f16.
pub fn avg_pool2d_backward_f16(
    x: FlexTensor,
    grad: FlexTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
) -> FlexTensor {
    let x_3d = expand_2d_to_3d(&x);
    let grad_3d = expand_2d_to_3d(&grad);
    let result = avg_pool3d_backward_f16(
        x_3d,
        grad_3d,
        [1, kernel_size[0], kernel_size[1]],
        [1, stride[0], stride[1]],
        [0, padding[0], padding[1]],
        count_include_pad,
    );
    squeeze_3d_to_2d(result)
}

/// Avg pool 2D backward for bf16.
pub fn avg_pool2d_backward_bf16(
    x: FlexTensor,
    grad: FlexTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
) -> FlexTensor {
    let x_f32 = convert_bf16_to_f32(&x);
    let grad_f32 = convert_bf16_to_f32(&grad);
    let result_f32 = avg_pool2d_backward_f32(
        x_f32,
        grad_f32,
        kernel_size,
        stride,
        padding,
        count_include_pad,
    );
    convert_f32_to_bf16(&result_f32)
}

avg_pool3d_backward_typed!(
    avg_pool3d_backward_f32,
    f32,
    DType::F32,
    0.0f32,
    |a, b| a + b,
    |val, count| val / count as f32
);
avg_pool3d_backward_typed!(
    avg_pool3d_backward_f64,
    f64,
    DType::F64,
    0.0f64,
    |a, b| a + b,
    |val, count| val / count as f64
);
avg_pool3d_backward_typed!(
    avg_pool3d_backward_f16,
    f16,
    DType::F16,
    f16::from_f32(0.0),
    |a: f16, b: f16| f16::from_f32(a.to_f32() + b.to_f32()),
    |val: f16, count| f16::from_f32(val.to_f32() / count as f32)
);

/// Generic avg pool 3D backward implementation.
#[allow(clippy::too_many_arguments)]
fn avg_pool3d_backward_impl<T>(
    x: FlexTensor,
    grad: FlexTensor,
    kernel_size: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
    count_include_pad: bool,
    dtype: DType,
    zero: T,
    add_fn: fn(T, T) -> T,
    div_fn: fn(T, usize) -> T,
) -> FlexTensor
where
    T: bytemuck::Pod + Copy + Send + Sync + Element,
{
    let x_shape = x.layout().shape();
    let grad = grad.to_contiguous();

    let batch_size = x_shape[0];
    let channels = x_shape[1];
    let in_d = x_shape[2];
    let in_h = x_shape[3];
    let in_w = x_shape[4];
    let spatial_in = in_d * in_h * in_w;

    let [kernel_d, kernel_h, kernel_w] = kernel_size;
    let [stride_d, stride_h, stride_w] = stride;
    let [pad_d, pad_h, pad_w] = padding;
    let kernel_volume = kernel_d * kernel_h * kernel_w;

    let grad_shape = grad.layout().shape();
    let out_d = grad_shape[2];
    let out_h = grad_shape[3];
    let out_w = grad_shape[4];
    let spatial_out = out_d * out_h * out_w;

    let grad_data: &[T] = grad.storage();

    // Distribute gradient equally across window
    let mut output = vec![zero; batch_size * channels * spatial_in];

    for b in 0..batch_size {
        for c in 0..channels {
            let grad_offset = b * channels * spatial_out + c * spatial_out;
            let out_offset = b * channels * spatial_in + c * spatial_in;

            for od in 0..out_d {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let grad_idx = grad_offset + od * out_h * out_w + oh * out_w + ow;
                        let grad_val = grad_data[grad_idx];

                        // Count valid positions for this output
                        let mut count = 0usize;
                        for kd in 0..kernel_d {
                            let id = (od * stride_d + kd) as isize - pad_d as isize;
                            if id >= 0 && id < in_d as isize {
                                for kh in 0..kernel_h {
                                    let ih = (oh * stride_h + kh) as isize - pad_h as isize;
                                    if ih >= 0 && ih < in_h as isize {
                                        for kw in 0..kernel_w {
                                            let iw = (ow * stride_w + kw) as isize - pad_w as isize;
                                            if iw >= 0 && iw < in_w as isize {
                                                count += 1;
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        let divisor = if count_include_pad {
                            kernel_volume
                        } else {
                            count.max(1)
                        };

                        // Distribute gradient
                        let distributed = div_fn(grad_val, divisor);
                        for kd in 0..kernel_d {
                            let id = (od * stride_d + kd) as isize - pad_d as isize;
                            if id < 0 || id >= in_d as isize {
                                continue;
                            }
                            let id = id as usize;

                            for kh in 0..kernel_h {
                                let ih = (oh * stride_h + kh) as isize - pad_h as isize;
                                if ih < 0 || ih >= in_h as isize {
                                    continue;
                                }
                                let ih = ih as usize;

                                for kw in 0..kernel_w {
                                    let iw = (ow * stride_w + kw) as isize - pad_w as isize;
                                    if iw < 0 || iw >= in_w as isize {
                                        continue;
                                    }
                                    let iw = iw as usize;

                                    let input_idx = out_offset + id * in_h * in_w + ih * in_w + iw;
                                    output[input_idx] = add_fn(output[input_idx], distributed);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let out_shape = Shape::from(vec![batch_size, channels, in_d, in_h, in_w]);
    FlexTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(out_shape),
        dtype,
    )
}

/// Adaptive avg pool 2D backward.
pub fn adaptive_avg_pool2d_backward_f32(x: FlexTensor, grad: FlexTensor) -> FlexTensor {
    let x_3d = expand_2d_to_3d(&x);
    let grad_3d = expand_2d_to_3d(&grad);
    let result = adaptive_avg_pool3d_backward_f32(x_3d, grad_3d);
    squeeze_3d_to_2d(result)
}

/// Adaptive avg pool 2D backward for f64.
pub fn adaptive_avg_pool2d_backward_f64(x: FlexTensor, grad: FlexTensor) -> FlexTensor {
    let x_3d = expand_2d_to_3d(&x);
    let grad_3d = expand_2d_to_3d(&grad);
    let result = adaptive_avg_pool3d_backward_f64(x_3d, grad_3d);
    squeeze_3d_to_2d(result)
}

/// Adaptive avg pool 2D backward for f16.
pub fn adaptive_avg_pool2d_backward_f16(x: FlexTensor, grad: FlexTensor) -> FlexTensor {
    let x_3d = expand_2d_to_3d(&x);
    let grad_3d = expand_2d_to_3d(&grad);
    let result = adaptive_avg_pool3d_backward_f16(x_3d, grad_3d);
    squeeze_3d_to_2d(result)
}

/// Adaptive avg pool 2D backward for bf16.
pub fn adaptive_avg_pool2d_backward_bf16(x: FlexTensor, grad: FlexTensor) -> FlexTensor {
    let x_f32 = convert_bf16_to_f32(&x);
    let grad_f32 = convert_bf16_to_f32(&grad);
    let result_f32 = adaptive_avg_pool2d_backward_f32(x_f32, grad_f32);
    convert_f32_to_bf16(&result_f32)
}

adaptive_avg_pool3d_backward_typed!(
    adaptive_avg_pool3d_backward_f32,
    f32,
    DType::F32,
    0.0f32,
    |a, b| a + b,
    |val, count| val / count as f32
);
adaptive_avg_pool3d_backward_typed!(
    adaptive_avg_pool3d_backward_f64,
    f64,
    DType::F64,
    0.0f64,
    |a, b| a + b,
    |val, count| val / count as f64
);
adaptive_avg_pool3d_backward_typed!(
    adaptive_avg_pool3d_backward_f16,
    f16,
    DType::F16,
    f16::from_f32(0.0),
    |a: f16, b: f16| f16::from_f32(a.to_f32() + b.to_f32()),
    |val: f16, count| f16::from_f32(val.to_f32() / count as f32)
);

/// Generic adaptive avg pool 3D backward implementation.
fn adaptive_avg_pool3d_backward_impl<T>(
    x: FlexTensor,
    grad: FlexTensor,
    dtype: DType,
    zero: T,
    add_fn: fn(T, T) -> T,
    div_fn: fn(T, usize) -> T,
) -> FlexTensor
where
    T: bytemuck::Pod + Copy + Send + Sync + Element,
{
    let x_shape = x.layout().shape();
    let grad = grad.to_contiguous();

    let batch_size = x_shape[0];
    let channels = x_shape[1];
    let in_d = x_shape[2];
    let in_h = x_shape[3];
    let in_w = x_shape[4];
    let spatial_in = in_d * in_h * in_w;

    let grad_shape = grad.layout().shape();
    let out_d = grad_shape[2];
    let out_h = grad_shape[3];
    let out_w = grad_shape[4];
    let spatial_out = out_d * out_h * out_w;

    let grad_data: &[T] = grad.storage();

    let mut output = vec![zero; batch_size * channels * spatial_in];

    for b in 0..batch_size {
        for c in 0..channels {
            let grad_offset = b * channels * spatial_out + c * spatial_out;
            let out_offset = b * channels * spatial_in + c * spatial_in;

            for od in 0..out_d {
                let d_start = (od * in_d) / out_d;
                let d_end = ((od + 1) * in_d).div_ceil(out_d);

                for oh in 0..out_h {
                    let h_start = (oh * in_h) / out_h;
                    let h_end = ((oh + 1) * in_h).div_ceil(out_h);

                    for ow in 0..out_w {
                        let w_start = (ow * in_w) / out_w;
                        let w_end = ((ow + 1) * in_w).div_ceil(out_w);

                        let grad_idx = grad_offset + od * out_h * out_w + oh * out_w + ow;
                        let grad_val = grad_data[grad_idx];

                        let count = (d_end - d_start) * (h_end - h_start) * (w_end - w_start);
                        let distributed = div_fn(grad_val, count.max(1));

                        for id in d_start..d_end {
                            for ih in h_start..h_end {
                                for iw in w_start..w_end {
                                    let input_idx = out_offset + id * in_h * in_w + ih * in_w + iw;
                                    output[input_idx] = add_fn(output[input_idx], distributed);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let out_shape = Shape::from(vec![batch_size, channels, in_d, in_h, in_w]);
    FlexTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(out_shape),
        dtype,
    )
}

// ============================================================================
// Dimension expansion/squeeze helpers
// ============================================================================

/// Expand 2D tensor [N, C, H, W] to 3D [N, C, 1, H, W].
fn expand_2d_to_3d(x: &FlexTensor) -> FlexTensor {
    let shape = x.layout().shape();
    x.reshape(Shape::from(vec![shape[0], shape[1], 1, shape[2], shape[3]]))
}

/// Squeeze 3D tensor [N, C, 1, H, W] to 2D [N, C, H, W].
fn squeeze_3d_to_2d(x: FlexTensor) -> FlexTensor {
    let shape = x.layout().shape();
    x.reshape(Shape::from(vec![shape[0], shape[1], shape[3], shape[4]]))
}

/// Expand 1D tensor [N, C, L] to 3D [N, C, 1, 1, L].
fn expand_1d_to_3d(x: &FlexTensor) -> FlexTensor {
    let shape = x.layout().shape();
    x.reshape(Shape::from(vec![shape[0], shape[1], 1, 1, shape[2]]))
}

/// Squeeze 3D tensor [N, C, 1, 1, L] to 1D [N, C, L].
fn squeeze_3d_to_1d(x: FlexTensor) -> FlexTensor {
    let shape = x.layout().shape();
    x.reshape(Shape::from(vec![shape[0], shape[1], shape[4]]))
}

// ============================================================================
// bf16 conversion helpers
// ============================================================================

fn convert_bf16_to_f32(tensor: &FlexTensor) -> FlexTensor {
    let tensor = tensor.to_contiguous();
    let data: &[bf16] = tensor.storage();
    let f32_data: Vec<f32> = data.iter().map(|x| x.to_f32()).collect();
    FlexTensor::new(
        Bytes::from_elems(f32_data),
        Layout::contiguous(tensor.layout().shape().clone()),
        DType::F32,
    )
}

fn convert_f32_to_bf16(tensor: &FlexTensor) -> FlexTensor {
    let data: &[f32] = tensor.storage();
    let bf16_data: Vec<bf16> = data.iter().map(|x| bf16::from_f32(*x)).collect();
    FlexTensor::new(
        Bytes::from_elems(bf16_data),
        Layout::contiguous(tensor.layout().shape().clone()),
        DType::BF16,
    )
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::TensorData;

    #[test]
    fn test_pool_output_size() {
        // Basic: input=4, kernel=2, padding=0, stride=2, dilation=1
        // output = (4 + 0 - 2) / 2 + 1 = 2
        assert_eq!(pool_output_size(4, 2, 0, 2, 1, false), 2);

        // With padding: input=4, kernel=2, padding=1, stride=2
        // output = (4 + 2 - 2) / 2 + 1 = 3
        assert_eq!(pool_output_size(4, 2, 1, 2, 1, false), 3);

        // With ceil mode: input=5, kernel=2, padding=0, stride=2
        // floor: (5 - 2) / 2 + 1 = 2
        // ceil: ceil(3/2) + 1 = 3
        assert_eq!(pool_output_size(5, 2, 0, 2, 1, false), 2);
        assert_eq!(pool_output_size(5, 2, 0, 2, 1, true), 3);

        // With dilation: input=7, kernel=2, dilation=2
        // effective_kernel = 2*(2-1)+1 = 3
        // output = (7 - 3) / 1 + 1 = 5
        assert_eq!(pool_output_size(7, 2, 0, 1, 2, false), 5);
    }

    #[test]
    fn test_max_pool2d_basic() {
        // Input: 4x4, kernel 2x2, stride 2
        // Input values: 1..16
        let x_data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let x = FlexTensor::from_data(TensorData::new(x_data, vec![1, 1, 4, 4]));

        let result = max_pool2d_f32(x, [2, 2], [2, 2], [0, 0], [1, 1], false);
        assert_eq!(result.layout().shape().to_vec(), vec![1, 1, 2, 2]);

        let out: Vec<f32> = result.into_data().to_vec().unwrap();
        // Max in each 2x2 block:
        // [1,2,3,4]   -> max(1,2,5,6)=6,  max(3,4,7,8)=8
        // [5,6,7,8]
        // [9,10,11,12] -> max(9,10,13,14)=14, max(11,12,15,16)=16
        // [13,14,15,16]
        assert_eq!(out, vec![6.0, 8.0, 14.0, 16.0]);
    }

    #[test]
    fn test_max_pool2d_with_indices() {
        let x_data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let x = FlexTensor::from_data(TensorData::new(x_data, vec![1, 1, 4, 4]));

        let (output, indices) =
            max_pool2d_with_indices_f32(x, [2, 2], [2, 2], [0, 0], [1, 1], false);

        let out: Vec<f32> = output.into_data().to_vec().unwrap();
        let idx: Vec<i64> = indices.into_data().to_vec().unwrap();

        assert_eq!(out, vec![6.0, 8.0, 14.0, 16.0]);
        // Indices into input spatial dims (flattened)
        // 6 is at (1,1) = 1*4+1 = 5
        // 8 is at (1,3) = 1*4+3 = 7
        // 14 is at (3,1) = 3*4+1 = 13
        // 16 is at (3,3) = 3*4+3 = 15
        assert_eq!(idx, vec![5, 7, 13, 15]);
    }

    #[test]
    fn test_max_pool2d_with_padding() {
        let x_data: Vec<f32> = (1..=9).map(|x| x as f32).collect();
        let x = FlexTensor::from_data(TensorData::new(x_data, vec![1, 1, 3, 3]));

        let result = max_pool2d_f32(x, [2, 2], [1, 1], [1, 1], [1, 1], false);
        assert_eq!(result.layout().shape().to_vec(), vec![1, 1, 4, 4]);
    }

    #[test]
    fn test_max_pool2d_with_dilation() {
        // 5x5 input, 2x2 kernel, dilation 2
        let x_data: Vec<f32> = (1..=25).map(|x| x as f32).collect();
        let x = FlexTensor::from_data(TensorData::new(x_data, vec![1, 1, 5, 5]));

        let result = max_pool2d_f32(x, [2, 2], [1, 1], [0, 0], [2, 2], false);
        // Effective kernel size = (2-1)*2 + 1 = 3
        // Output size = (5 - 3) / 1 + 1 = 3
        assert_eq!(result.layout().shape().to_vec(), vec![1, 1, 3, 3]);

        let out: Vec<f32> = result.into_data().to_vec().unwrap();
        // With dilation 2, kernel at (0,0) hits positions (0,0), (0,2), (2,0), (2,2)
        // Input layout (1-indexed for readability):
        // 1  2  3  4  5
        // 6  7  8  9  10
        // 11 12 13 14 15
        // 16 17 18 19 20
        // 21 22 23 24 25
        // First output at (0,0): max(1, 3, 11, 13) = 13
        assert_eq!(out[0], 13.0);
    }

    #[test]
    fn test_max_pool2d_ceil_mode() {
        let x_data: Vec<f32> = (1..=25).map(|x| x as f32).collect();
        let x = FlexTensor::from_data(TensorData::new(x_data, vec![1, 1, 5, 5]));

        // Without ceil mode: (5 - 2) / 2 + 1 = 2
        let result_floor = max_pool2d_f32(x.clone(), [2, 2], [2, 2], [0, 0], [1, 1], false);
        assert_eq!(result_floor.layout().shape().to_vec(), vec![1, 1, 2, 2]);

        // With ceil mode: ceil((5 - 2) / 2) + 1 = 3
        let result_ceil = max_pool2d_f32(x, [2, 2], [2, 2], [0, 0], [1, 1], true);
        assert_eq!(result_ceil.layout().shape().to_vec(), vec![1, 1, 3, 3]);
    }

    #[test]
    fn test_avg_pool2d_basic() {
        let x_data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let x = FlexTensor::from_data(TensorData::new(x_data, vec![1, 1, 4, 4]));

        let result = avg_pool2d_f32(x, [2, 2], [2, 2], [0, 0], false, false);
        assert_eq!(result.layout().shape().to_vec(), vec![1, 1, 2, 2]);

        let out: Vec<f32> = result.into_data().to_vec().unwrap();
        // Avg in each 2x2 block:
        // (1+2+5+6)/4 = 3.5, (3+4+7+8)/4 = 5.5
        // (9+10+13+14)/4 = 11.5, (11+12+15+16)/4 = 13.5
        assert!((out[0] - 3.5).abs() < 1e-5);
        assert!((out[1] - 5.5).abs() < 1e-5);
        assert!((out[2] - 11.5).abs() < 1e-5);
        assert!((out[3] - 13.5).abs() < 1e-5);
    }

    #[test]
    fn test_avg_pool2d_count_include_pad() {
        // 3x3 input with padding 1, kernel 2x2, stride 2
        let x_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let x = FlexTensor::from_data(TensorData::new(x_data, vec![1, 1, 3, 3]));

        // count_include_pad = true: divide by kernel size (4)
        let result_include = avg_pool2d_f32(x.clone(), [2, 2], [2, 2], [1, 1], true, false);
        let out_include: Vec<f32> = result_include.into_data().to_vec().unwrap();

        // count_include_pad = false: divide by actual count
        let result_exclude = avg_pool2d_f32(x, [2, 2], [2, 2], [1, 1], false, false);
        let out_exclude: Vec<f32> = result_exclude.into_data().to_vec().unwrap();

        // Corner position with padding: only 1 valid element
        // With count_include_pad: 1.0 / 4 = 0.25
        // Without: 1.0 / 1 = 1.0
        assert!((out_include[0] - 0.25).abs() < 1e-5);
        assert!((out_exclude[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_adaptive_avg_pool2d() {
        // 4x4 input -> 2x2 output
        let x_data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let x = FlexTensor::from_data(TensorData::new(x_data, vec![1, 1, 4, 4]));

        let result = adaptive_avg_pool2d_f32(x, [2, 2]);
        assert_eq!(result.layout().shape().to_vec(), vec![1, 1, 2, 2]);

        let out: Vec<f32> = result.into_data().to_vec().unwrap();
        // Each output averages a 2x2 region
        assert!((out[0] - 3.5).abs() < 1e-5); // (1+2+5+6)/4
        assert!((out[1] - 5.5).abs() < 1e-5); // (3+4+7+8)/4
    }

    #[test]
    fn test_adaptive_avg_pool2d_global() {
        // Global pooling: any size -> 1x1
        let x_data: Vec<f32> = (1..=9).map(|x| x as f32).collect();
        let x = FlexTensor::from_data(TensorData::new(x_data, vec![1, 1, 3, 3]));

        let result = adaptive_avg_pool2d_f32(x, [1, 1]);
        assert_eq!(result.layout().shape().to_vec(), vec![1, 1, 1, 1]);

        let out: Vec<f32> = result.into_data().to_vec().unwrap();
        // Average of 1..9 = 5
        assert!((out[0] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_pool1d_delegation() {
        // 1D pooling should work via 3D delegation
        let x_data: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let x = FlexTensor::from_data(TensorData::new(x_data, vec![1, 1, 8]));

        let result = max_pool1d_f32(x, 2, 2, 0, 1, false);
        assert_eq!(result.layout().shape().to_vec(), vec![1, 1, 4]);

        let out: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(out, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_max_pool3d_basic() {
        // Simple 3D pooling test: [1, 2, 4, 4, 4] = 1*2*4*4*4 = 128 elements
        let x_data = vec![1.0f32; 1 * 2 * 4 * 4 * 4];
        let x = FlexTensor::from_data(TensorData::new(x_data, vec![1, 2, 4, 4, 4]));

        let result = max_pool3d_f32(x, [2, 2, 2], [2, 2, 2], [0, 0, 0], [1, 1, 1], false);
        assert_eq!(result.layout().shape().to_vec(), vec![1, 2, 2, 2, 2]);
    }

    #[test]
    fn test_max_pool2d_f64() {
        let x_data: Vec<f64> = (1..=16).map(|x| x as f64).collect();
        let x = FlexTensor::from_data(TensorData::new(x_data, vec![1, 1, 4, 4]));

        let result = max_pool2d_f64(x, [2, 2], [2, 2], [0, 0], [1, 1], false);
        let out: Vec<f64> = result.into_data().to_vec().unwrap();
        assert_eq!(out, vec![6.0, 8.0, 14.0, 16.0]);
    }

    #[test]
    fn test_max_pool2d_f16() {
        let x_data: Vec<f16> = (1..=16).map(|x| f16::from_f32(x as f32)).collect();
        let x = FlexTensor::from_data(TensorData::new(x_data, vec![1, 1, 4, 4]));

        let result = max_pool2d_f16(x, [2, 2], [2, 2], [0, 0], [1, 1], false);
        let out: Vec<f16> = result.into_data().to_vec().unwrap();

        assert!((out[0].to_f32() - 6.0).abs() < 0.1);
        assert!((out[1].to_f32() - 8.0).abs() < 0.1);
        assert!((out[2].to_f32() - 14.0).abs() < 0.1);
        assert!((out[3].to_f32() - 16.0).abs() < 0.1);
    }

    #[test]
    fn test_max_pool2d_bf16() {
        let x_data: Vec<bf16> = (1..=16).map(|x| bf16::from_f32(x as f32)).collect();
        let x = FlexTensor::from_data(TensorData::new(x_data, vec![1, 1, 4, 4]));

        let result = max_pool2d_bf16(x, [2, 2], [2, 2], [0, 0], [1, 1], false);
        let out: Vec<bf16> = result.into_data().to_vec().unwrap();

        assert!((out[0].to_f32() - 6.0).abs() < 0.5);
        assert!((out[1].to_f32() - 8.0).abs() < 0.5);
    }

    #[test]
    fn test_max_pool_backward() {
        // Forward pass
        let x_data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let x = FlexTensor::from_data(TensorData::new(x_data.clone(), vec![1, 1, 4, 4]));
        let (_output, indices) =
            max_pool2d_with_indices_f32(x.clone(), [2, 2], [2, 2], [0, 0], [1, 1], false);

        // Backward pass: gradient of 1.0 for each output
        let grad = FlexTensor::from_data(TensorData::new(vec![1.0f32; 4], vec![1, 1, 2, 2]));

        let x_grad = max_pool2d_backward_f32(x, grad, indices);
        let grad_data: Vec<f32> = x_grad.into_data().to_vec().unwrap();

        // Gradient should be 1.0 at max positions, 0.0 elsewhere
        // Max positions were: 5, 7, 13, 15 (0-indexed)
        assert_eq!(grad_data[5], 1.0);
        assert_eq!(grad_data[7], 1.0);
        assert_eq!(grad_data[13], 1.0);
        assert_eq!(grad_data[15], 1.0);
        assert_eq!(grad_data[0], 0.0);
    }

    #[test]
    fn test_avg_pool_backward() {
        let x_data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let x = FlexTensor::from_data(TensorData::new(x_data, vec![1, 1, 4, 4]));

        // Backward with gradient of 4.0 for each output (will distribute as 1.0 each)
        let grad = FlexTensor::from_data(TensorData::new(vec![4.0f32; 4], vec![1, 1, 2, 2]));

        let x_grad = avg_pool2d_backward_f32(x, grad, [2, 2], [2, 2], [0, 0], false);
        let grad_data: Vec<f32> = x_grad.into_data().to_vec().unwrap();

        // Each position in input should receive grad/4 = 1.0
        assert!(grad_data.iter().all(|&v| (v - 1.0).abs() < 1e-5));
    }

    #[test]
    fn test_adaptive_avg_pool_backward() {
        let x_data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let x = FlexTensor::from_data(TensorData::new(x_data, vec![1, 1, 4, 4]));

        // Backward with gradient of 4.0 for each output element
        let grad = FlexTensor::from_data(TensorData::new(vec![4.0f32; 4], vec![1, 1, 2, 2]));
        let x_grad = adaptive_avg_pool2d_backward_f32(x, grad);
        let grad_data: Vec<f32> = x_grad.into_data().to_vec().unwrap();

        // Each input position receives gradient from its output region
        assert!(grad_data.iter().all(|&v| (v - 1.0).abs() < 1e-5));
    }

    #[test]
    #[should_panic(expected = "kernel size must be > 0")]
    fn test_pool_output_size_zero_kernel_panics() {
        pool_output_size(4, 0, 0, 1, 1, false);
    }

    #[test]
    #[should_panic(expected = "stride must be > 0")]
    fn test_pool_output_size_zero_stride_panics() {
        pool_output_size(4, 2, 0, 0, 1, false);
    }
}
