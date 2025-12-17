use burn_backend::ops::ConvOptions;
use burn_std::Shape;
use cubek::convolution::{AcceleratedTileKind, components::ConvSetupError};

#[cfg(feature = "autotune")]
use crate::kernel::conv::wgrad_autotune;
use crate::{
    CubeRuntime,
    kernel::conv::{
        backward_weight::implicit_gemm::wgrad_gemm_simple_sync,
        fallback::conv_weight_backward_fallback, forward::implicit_gemm::conv_gemm_simple_sync,
    },
    ops::{permute_nchw_to_nhwc, permute_nchw_to_nhwc_shape, permute_nhwc_to_nchw},
    tensor::CubeTensor,
};

use super::conv_direct;
#[cfg(feature = "autotune")]
use super::forward::conv_autotune;

/// The strategy to be used when launching a convolution kernel.
pub enum ConvStrategy {
    /// A simple direct convolution.
    Direct,
    #[cfg(feature = "autotune")]
    /// Using autotune to choose the best kernel based on runtime information.
    Autotune,
    /// Implicit GEMM implementation of convolution. Lower memory usage but requires CMMA and
    /// has constraints on tensor shape.
    ImplicitGemm,
}

impl Default for ConvStrategy {
    fn default() -> Self {
        // if autotune is enabled, default to autotune
        #[cfg(feature = "autotune")]
        return ConvStrategy::Autotune;

        // if autotune is disabled, default to the more memory-conservative algorithm
        #[cfg(not(feature = "autotune"))]
        ConvStrategy::Direct
    }
}

/// Performs an N-dimensional convolution with the given strategy
///
/// * `input` - The input feature map
/// * `weight` - The weights (filter) applied to each kernel
/// * `bias` - The bias added to each channel
/// * `options` - The options to use for the convolution
/// * `strategy` - The convolution algorithm to use. Autotune will pick the fastest available option.
pub fn conv_forward<R: CubeRuntime, const N: usize>(
    input: CubeTensor<R>,
    weight: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    options: ConvOptions<N>,
    strategy: ConvStrategy,
) -> Result<CubeTensor<R>, ConvSetupError> {
    let input = permute_nchw_to_nhwc(input);
    let weight = permute_nchw_to_nhwc(weight);

    let out = conv_forward_nhwc(input, weight, bias, options, strategy)?;

    Ok(permute_nhwc_to_nchw(out))
}

pub fn conv_forward_nhwc<R: CubeRuntime, const N: usize>(
    input: CubeTensor<R>,
    weight: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    options: ConvOptions<N>,
    strategy: ConvStrategy,
) -> Result<CubeTensor<R>, ConvSetupError> {
    match strategy {
        ConvStrategy::Direct => conv_direct::<R, N>(input, weight, bias, options),
        #[cfg(feature = "autotune")]
        ConvStrategy::Autotune => Ok(conv_autotune::<R, N>(input, weight, bias, options)),
        ConvStrategy::ImplicitGemm => {
            if options.groups != 1 {
                conv_direct::<R, N>(input, weight, bias, options)
            } else {
                conv_gemm_simple_sync::<R, N>(
                    input,
                    weight,
                    bias,
                    options,
                    AcceleratedTileKind::Cmma,
                )
            }
        }
    }
}

/// Performs an N-dimensional convolution backwards pass with regard to weight, with the given strategy
///
/// * `input` - The input feature map
/// * `out_grad` - The output gradients
/// * `weight_shape` - The shape of the weights/weight gradients
/// * `options` - The options used for the convolution
/// * `strategy` - The convolution algorithm to use. Autotune will pick the fastest available option.
pub fn conv_weight_backward<R: CubeRuntime, const N: usize>(
    input: CubeTensor<R>,
    out_grad: CubeTensor<R>,
    weight_shape: Shape,
    options: ConvOptions<N>,
    strategy: ConvStrategy,
) -> Result<CubeTensor<R>, ConvSetupError> {
    let input = permute_nchw_to_nhwc(input);
    let out_grad = permute_nchw_to_nhwc(out_grad);
    let weight_shape = permute_nchw_to_nhwc_shape(weight_shape);

    let weight_grad = match strategy {
        ConvStrategy::Direct => {
            conv_weight_backward_fallback::<R, N>(input, out_grad, weight_shape, options)
        }
        #[cfg(feature = "autotune")]
        ConvStrategy::Autotune => Ok(wgrad_autotune::<R, N>(
            input,
            out_grad,
            weight_shape,
            options,
        )),
        ConvStrategy::ImplicitGemm => {
            if options.groups != 1 {
                conv_weight_backward_fallback::<R, N>(input, out_grad, weight_shape, options)
            } else {
                wgrad_gemm_simple_sync::<R, N>(
                    input,
                    out_grad,
                    weight_shape,
                    options,
                    AcceleratedTileKind::Cmma,
                )
            }
        }
    }?;

    Ok(permute_nhwc_to_nchw(weight_grad))
}
