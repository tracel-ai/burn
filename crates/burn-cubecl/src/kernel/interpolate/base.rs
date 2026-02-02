use crate::{
    CubeRuntime,
    kernel::into_contiguous,
    ops::{numeric::empty_device_dtype, permute_nchw_to_nhwc, permute_nhwc_to_nchw},
    tensor::CubeTensor,
};
use burn_backend::{
    Shape,
    ops::{InterpolateMode, InterpolateOptions},
};

use super::{
    bicubic::interpolate_bicubic_launch, bilinear::interpolate_bilinear_launch,
    nearest::interpolate_nearest_launch, nearest_backward::interpolate_nearest_backward_launch,
};

/// Interpolate operation
///
/// Supports nearest, bilinear and bicubic modes
pub fn interpolate<R: CubeRuntime>(
    input: CubeTensor<R>,
    output_size: [usize; 2],
    options: InterpolateOptions,
) -> CubeTensor<R> {
    let [batch_size, channels, _, _] = input.shape.dims();
    let [out_height, out_width] = output_size;

    let input = into_contiguous(permute_nchw_to_nhwc(input));

    let shape_out = Shape::new([batch_size, out_height, out_width, channels]);
    let output = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        shape_out,
        input.dtype,
    );

    let output = match options.mode {
        InterpolateMode::Nearest => interpolate_nearest_launch(input, output),
        InterpolateMode::Bilinear => interpolate_bilinear_launch(input, output),
        InterpolateMode::Bicubic => interpolate_bicubic_launch(input, output),
    };

    permute_nhwc_to_nchw(output)
}

/// Backward interpolate operation
///
/// Note: only nearest mode is supported
pub fn interpolate_backward<R: CubeRuntime>(
    input: CubeTensor<R>,
    out_grad: CubeTensor<R>,
    _output_size: [usize; 2],
    options: InterpolateOptions,
) -> CubeTensor<R> {
    let input = permute_nchw_to_nhwc(input);
    let out_grad = permute_nchw_to_nhwc(out_grad);

    let output_shape = input.shape.clone();
    let output = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        output_shape,
        input.dtype,
    );

    let output = match options.mode {
        InterpolateMode::Nearest => interpolate_nearest_backward_launch(out_grad, output),
        InterpolateMode::Bilinear => {
            panic!("bilinear interpolation backward is not supported by JIT backend")
        }
        InterpolateMode::Bicubic => {
            panic!("bicubic interpolation backward is not supported by JIT backend")
        }
    };

    permute_nhwc_to_nchw(output)
}
