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
#[cfg(not(feature = "autotune"))]
use cubek::interpolate::Residence;
use cubek::interpolate::{
    InterpolateConfig,
    definition::{
        InterpolateError, InterpolateMode as CubekInterpolateMode,
        InterpolateOptions as CubekInterpolateOptions, NearestMode as CubekNearestMode,
    },
    interpolate as cubek_interpolate, interpolate_backward as cubek_interpolate_backward,
};

#[derive(Debug)]
/// Strategy used to select how interpolation runs.
pub enum InterpolateStrategy {
    /// Run the kernel with the given geometry.
    Config(InterpolateConfig),

    /// Automatically benchmark and select the best configuration at runtime.
    #[cfg(feature = "autotune")]
    Autotune,
}

impl Default for InterpolateStrategy {
    fn default() -> Self {
        // if autotune is enabled, default to autotune
        #[cfg(feature = "autotune")]
        return InterpolateStrategy::Autotune;

        // A guess, not a measurement: it sits mid-box in cubek's benchmark catalogue and has
        // no recorded sweep behind it. Replace it once cubek publishes one.
        #[cfg(not(feature = "autotune"))]
        InterpolateStrategy::Config(InterpolateConfig::new(Residence::InPlace, 4, 2, 1))
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
        InterpolateStrategy::Config(config) => {
            execute_interpolate(input, output_size, options, config)
        }
        #[cfg(feature = "autotune")]
        InterpolateStrategy::Autotune => Ok(interpolate_autotune(input, output_size, options)),
    }
}

/// Execute interpolation with the given configuration, without autotuning. This is used by the
/// autotune implementation to run each candidate configuration.
pub fn execute_interpolate<R: CubeRuntime>(
    input: CubeTensor<R>,
    output_size: [usize; 2],
    options: InterpolateOptions,
    config: InterpolateConfig,
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
        config,
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
