use crate::{
    element::JitElement, kernel::into_contiguous, ops::numeric::empty_device, tensor::JitTensor,
    Runtime,
};
use burn_tensor::{
    ops::{InterpolateMode, InterpolateOptions},
    Element, Shape,
};

use super::{
    bicubic::interpolate_bicubic_launch, bilinear::interpolate_bilinear_launch,
    nearest::interpolate_nearest_launch, nearest_backward::interpolate_nearest_backward_launch,
};

/// Interpolate operation
///
/// Supports nearest, bilinear and bicubic modes
pub fn interpolate<R: Runtime, E: JitElement + Element>(
    input: JitTensor<R, E, 4>,
    output_size: [usize; 2],
    options: InterpolateOptions,
) -> JitTensor<R, E, 4> {
    let input = into_contiguous(input);
    let [batch_size, channels, _, _] = input.shape.dims;
    let [out_height, out_width] = output_size;

    let shape_out = Shape::new([batch_size, channels, out_height, out_width]);
    let output = empty_device(input.client.clone(), input.device.clone(), shape_out);

    match options.mode {
        InterpolateMode::Nearest => interpolate_nearest_launch(input, output),
        InterpolateMode::Bilinear => interpolate_bilinear_launch(input, output),
        InterpolateMode::Bicubic => interpolate_bicubic_launch(input, output),
    }
}

/// Backward interpolate operation
///
/// Note: only nearest mode is supported
pub fn interpolate_backward<R: Runtime, E: JitElement + Element>(
    input: JitTensor<R, E, 4>,
    out_grad: JitTensor<R, E, 4>,
    _output_size: [usize; 2],
    options: InterpolateOptions,
) -> JitTensor<R, E, 4> {
    let out_grad = into_contiguous(out_grad);
    let output_shape = input.shape.clone();
    let num_elems = input.shape.num_elements();
    let buffer = input.client.empty(num_elems * core::mem::size_of::<E>());
    let output = JitTensor::new(
        input.client.clone(),
        input.device.clone(),
        output_shape,
        buffer,
    );

    match options.mode {
        InterpolateMode::Nearest => interpolate_nearest_backward_launch(out_grad, output),
        InterpolateMode::Bilinear => {
            panic!("bilinear interpolation backward is not supported by JIT backend")
        }
        InterpolateMode::Bicubic => {
            panic!("bicubic interpolation backward is not supported by JIT backend")
        }
    }
}
