//! Shared macros and helpers for forward and transposed convolution.

use alloc::vec::Vec;
use burn_backend::DType;
use burn_std::{Bytes, Shape, bf16};

use crate::{FlexTensor, Layout};

// ============================================================================
// Macros for dtype wrappers
// ============================================================================

/// Generates a convNd function that delegates to a conv3d function via expand/squeeze.
macro_rules! conv_nd_via_3d {
    ($fn_name:ident, $conv3d_fn:ident, $expand_fn:ident, $squeeze_fn:ident, $dim:literal, $Options:ident) => {
        pub fn $fn_name(
            x: FlexTensor,
            weight: FlexTensor,
            bias: Option<FlexTensor>,
            options: &$Options<$dim>,
        ) -> FlexTensor {
            let (x_3d, weight_3d, options_3d) = $expand_fn(&x, &weight, options);
            let result_3d = $conv3d_fn(x_3d, weight_3d, bias, &options_3d);
            $squeeze_fn(result_3d)
        }
    };
}

/// Generates a bf16 function that converts to f32, calls the f32 variant, converts back.
macro_rules! bf16_via_f32 {
    ($bf16_fn:ident, $f32_fn:ident, $dim:literal, $Options:ident) => {
        pub fn $bf16_fn(
            x: FlexTensor,
            weight: FlexTensor,
            bias: Option<FlexTensor>,
            options: &$Options<$dim>,
        ) -> FlexTensor {
            let x_f32 = $crate::ops::conv_common::convert_bf16_to_f32(&x);
            let weight_f32 = $crate::ops::conv_common::convert_bf16_to_f32(&weight);
            let bias_f32 = bias.map(|b| $crate::ops::conv_common::convert_bf16_to_f32(&b));
            let result_f32 = $f32_fn(x_f32, weight_f32, bias_f32, options);
            $crate::ops::conv_common::convert_f32_to_bf16(&result_f32)
        }
    };
}

// ============================================================================
// Squeeze helpers (used by both conv and conv_transpose 1d/2d paths)
// ============================================================================

pub(crate) fn squeeze_3d_to_1d(tensor: FlexTensor) -> FlexTensor {
    let shape = tensor.layout().shape();
    tensor.reshape(Shape::from(alloc::vec![shape[0], shape[1], shape[4]]))
}

pub(crate) fn squeeze_3d_to_2d(tensor: FlexTensor) -> FlexTensor {
    let shape = tensor.layout().shape();
    tensor.reshape(Shape::from(alloc::vec![
        shape[0], shape[1], shape[3], shape[4]
    ]))
}

// ============================================================================
// bf16 conversion helpers
// ============================================================================

pub(crate) fn convert_bf16_to_f32(tensor: &FlexTensor) -> FlexTensor {
    let tensor = tensor.to_contiguous();
    let data: &[bf16] = tensor.storage();
    let f32_data: Vec<f32> = data.iter().map(|x| x.to_f32()).collect();
    FlexTensor::new(
        Bytes::from_elems(f32_data),
        Layout::contiguous(tensor.layout().shape().clone()),
        DType::F32,
    )
}

pub(crate) fn convert_f32_to_bf16(tensor: &FlexTensor) -> FlexTensor {
    let data: &[f32] = tensor.storage();
    let bf16_data: Vec<bf16> = data.iter().map(|x| bf16::from_f32(*x)).collect();
    FlexTensor::new(
        Bytes::from_elems(bf16_data),
        Layout::contiguous(tensor.layout().shape().clone()),
        DType::BF16,
    )
}

// ============================================================================
// Bias addition
// ============================================================================

#[allow(clippy::needless_range_loop)]
pub(crate) fn add_bias<T: Copy>(
    output: &mut [T],
    bias: &[T],
    batch: usize,
    channels: usize,
    spatial: usize,
    add_fn: fn(T, T) -> T,
) {
    for b in 0..batch {
        for c in 0..channels {
            let offset = b * channels * spatial + c * spatial;
            let bias_val = bias[c];
            for i in 0..spatial {
                output[offset + i] = add_fn(output[offset + i], bias_val);
            }
        }
    }
}
