#[cfg(feature = "autotune")]
use crate::kernel::interpolate::interpolate_autotune;
use crate::{
    kernel::into_contiguous,
    ops::{numeric::empty_device_dtype, permute_nchw_to_nhwc, permute_nhwc_to_nchw},
    tensor::CubeTensor,
};
use burn_backend::cubecl::dtype_to_storage_type;
use burn_backend::{Shape, TensorMetadata, ops::InterpolateMode, ops::InterpolateOptions};
use cubek::interpolate::{
    InterpolateStrategy as CubekInterpolateStrategy,
    definition::{
        InterpolateError, InterpolateMode as CubekInterpolateMode,
        InterpolateOptions as CubekInterpolateOptions, NearestMode as CubekNearestMode,
    },
    interpolate as cubek_interpolate, interpolate_backward as cubek_interpolate_backward,
};

#[derive(Debug)]
/// Strategy used to select how interpolation runs.
pub enum InterpolateStrategy {
    /// Run the strategy given rather than searching for one. cubek resolves it against the device
    /// and the problem, so an intent states what the launch optimizes for and leaves the geometry
    /// to cubek, while [`Forced`](CubekInterpolateStrategy::Forced) pins the geometry outright.
    Specific(CubekInterpolateStrategy),

    /// Automatically benchmark and select the best strategy at runtime.
    #[cfg(feature = "autotune")]
    Autotune,
}

impl Default for InterpolateStrategy {
    fn default() -> Self {
        // if autotune is enabled, default to autotune
        #[cfg(feature = "autotune")]
        return InterpolateStrategy::Autotune;

        // Interpolation reads one tensor and writes another, so a build that measures nothing
        // runs the intent that takes memory for the limit.
        #[cfg(not(feature = "autotune"))]
        InterpolateStrategy::Specific(CubekInterpolateStrategy::MaximizeThroughput)
    }
}

/// Interpolate operation
///
/// Supports nearest, bilinear, bicubic and lanczos3 modes
pub fn interpolate(
    input: CubeTensor,
    output_size: [usize; 2],
    options: InterpolateOptions,
    strategy: InterpolateStrategy,
) -> Result<CubeTensor, InterpolateError> {
    match strategy {
        InterpolateStrategy::Specific(strategy) => {
            execute_interpolate(input, output_size, options, strategy)
        }
        #[cfg(feature = "autotune")]
        InterpolateStrategy::Autotune => Ok(interpolate_autotune(input, output_size, options)),
    }
}

/// Execute interpolation with the given strategy, without autotuning. This is used by the
/// autotune implementation to run each candidate strategy.
pub fn execute_interpolate(
    input: CubeTensor,
    output_size: [usize; 2],
    options: InterpolateOptions,
    strategy: CubekInterpolateStrategy,
) -> Result<CubeTensor, InterpolateError> {
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
        strategy,
        dtype_to_storage_type(input.dtype),
    )?;

    Ok(permute_nhwc_to_nchw(output))
}

/// Backward interpolate operation
///
/// Note: only nearest mode is supported
pub fn interpolate_backward(
    input: CubeTensor,
    out_grad: CubeTensor,
    _output_size: [usize; 2],
    options: InterpolateOptions,
) -> CubeTensor {
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
        dtype_to_storage_type(input.dtype),
    )
    .unwrap_or_else(|e| {
        panic!(
            "interpolate_backward kernel failed (device={0:?}, dtype={1:?}, options={2:?}): {3}",
            input.device, input.dtype, options, e
        )
    });

    permute_nhwc_to_nchw(output)
}

pub(crate) fn map_mode(mode: InterpolateMode) -> CubekInterpolateMode {
    match mode {
        InterpolateMode::Nearest => CubekInterpolateMode::Nearest(CubekNearestMode::Floor),
        InterpolateMode::NearestExact => CubekInterpolateMode::Nearest(CubekNearestMode::Exact),
        InterpolateMode::Bilinear => CubekInterpolateMode::Bilinear,
        InterpolateMode::Bicubic => CubekInterpolateMode::Bicubic,
        InterpolateMode::Lanczos3 => CubekInterpolateMode::Lanczos3,
    }
}

pub(crate) fn map_options(options: InterpolateOptions) -> CubekInterpolateOptions {
    CubekInterpolateOptions {
        mode: map_mode(options.mode),
        align_corners: options.align_corners,
    }
}
