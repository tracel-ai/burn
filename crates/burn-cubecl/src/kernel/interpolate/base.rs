use crate::{
    CubeRuntime,
    kernel::into_contiguous,
    ops::{numeric::empty_device_dtype, permute_nchw_to_nhwc, permute_nhwc_to_nchw},
    tensor::CubeTensor,
};
use burn_backend::{Shape, TensorMetadata, ops::InterpolateMode, ops::InterpolateOptions};
use cubek::interpolate::{
    definition::InterpolateMode as CubekInterpolateMode,
    definition::InterpolateOptions as CubekInterpolateOptions, interpolate as cubek_interpolate,
    interpolate_backward as cubek_interpolate_backward,
};

/// Interpolate operation
///
/// Supports nearest, bilinear, bicubic and lanczos3 modes
pub fn interpolate<R: CubeRuntime>(
    input: CubeTensor<R>,
    output_size: [usize; 2],
    options: InterpolateOptions,
) -> CubeTensor<R> {
    let [batch_size, channels, _, _] = input.meta.shape().dims();
    let [out_height, out_width] = output_size;

    let input = into_contiguous(permute_nchw_to_nhwc(input));

    let shape_out = Shape::new([batch_size, out_height, out_width, channels]);
    let output = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        shape_out,
        input.dtype,
    );

    cubek_interpolate(
        &input.client.clone(),
        input.clone().binding(),
        output.clone().binding(),
        map_options(options.clone()),
        input.dtype.into(),
    )
    .unwrap_or_else(|e| {
        panic!(
            "interpolate kernel failed (device={0:?}, dtype={1:?}, options={2:?}): {3}",
            input.device, input.dtype, options, e
        )
    });

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

    let output_shape = input.shape();
    let output = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        output_shape,
        input.dtype,
    );

    cubek_interpolate_backward(
        &input.client.clone(),
        input.clone().binding(),
        out_grad.binding(),
        output.clone().binding(),
        map_options(options.clone()),
        input.dtype.into(),
    )
    .unwrap_or_else(|e| {
        panic!(
            "interpolate_backward kernel failed (device={0:?}, dtype={1:?}, options={2:?}): {3}",
            input.device, input.dtype, options, e
        )
    });

    permute_nhwc_to_nchw(output)
}

fn map_options(options: InterpolateOptions) -> CubekInterpolateOptions {
    CubekInterpolateOptions {
        mode: {
            match options.mode {
                InterpolateMode::Nearest => CubekInterpolateMode::Nearest,
                InterpolateMode::Bilinear => CubekInterpolateMode::Bilinear,
                InterpolateMode::Bicubic => CubekInterpolateMode::Bicubic,
                InterpolateMode::Lanczos3 => CubekInterpolateMode::Lanczos3,
            }
        },
        align_corners: options.align_corners,
    }
}
