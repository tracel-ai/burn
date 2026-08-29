#[cfg(feature = "autotune")]
use crate::kernel::interpolate::interpolate_autotune;
use crate::{
    CubeRuntime,
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
/// Strategy used to select which interpolate implementation to run.
///
/// The two explicit variants are intents, not geometries: cubek resolves each one to a blueprint
/// from the device and the problem, so both are launchable everywhere and neither needs a tile
/// size stated here. They differ in how much of a cube one problem occupies and in whether the
/// gathered input is staged, which is the choice worth measuring.
pub enum InterpolateStrategy {
    /// Read the input where it lies and widen the cube, for a launch that waits on memory.
    MaximizeThroughput,

    /// Stage the gathered input so a window the taps re-read is fetched once, for a launch that
    /// waits on the tap window.
    MinimizeLatency,

    /// Automatically benchmark and select the best strategy at runtime.
    #[cfg(feature = "autotune")]
    Autotune,
}

impl Default for InterpolateStrategy {
    fn default() -> Self {
        // if autotune is enabled, default to autotune
        #[cfg(feature = "autotune")]
        return InterpolateStrategy::Autotune;

        // Without a measurement, take the intent that stages nothing: it is the one no device
        // declines for want of shared memory.
        #[cfg(not(feature = "autotune"))]
        InterpolateStrategy::MaximizeThroughput
    }
}

/// Interpolate operation
///
/// Supports nearest, bilinear, bicubic and lanczos3 modes
pub fn interpolate<R: CubeRuntime>(
    input: CubeTensor<R>,
    output_size: [usize; 2],
    options: InterpolateOptions,
    strategy: InterpolateStrategy,
) -> Result<CubeTensor<R>, InterpolateError> {
    match strategy {
        InterpolateStrategy::MaximizeThroughput => execute_interpolate(
            input,
            output_size,
            options,
            CubekInterpolateStrategy::MaximizeThroughput,
        ),
        InterpolateStrategy::MinimizeLatency => execute_interpolate(
            input,
            output_size,
            options,
            CubekInterpolateStrategy::MinimizeLatency,
        ),
        #[cfg(feature = "autotune")]
        InterpolateStrategy::Autotune => Ok(interpolate_autotune(input, output_size, options)),
    }
}

/// Execute the given interpolate strategy without autotuning. This is used by the autotune implementation to run each candidate strategy.
pub fn execute_interpolate<R: CubeRuntime>(
    input: CubeTensor<R>,
    output_size: [usize; 2],
    options: InterpolateOptions,
    strategy: CubekInterpolateStrategy,
) -> Result<CubeTensor<R>, InterpolateError> {
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
