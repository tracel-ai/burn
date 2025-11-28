use burn_tensor::ops::ConvOptions;
use cubecl::convolution::components::ConvSetupError;

use crate::{
    CubeRuntime,
    ops::{permute_nchw_to_nhwc, permute_nhwc_to_nchw},
    tensor::CubeTensor,
};

#[cfg(feature = "autotune")]
use super::conv_autotune;
use super::{conv_direct, conv_gemm_cyclic, conv_im2col};

/// The strategy to be used when launching a convolution kernel.
pub enum ConvStrategy {
    /// A simple direct convolution.
    Direct,
    #[cfg(feature = "autotune")]
    /// Using autotune to choose the best kernel based on runtime information.
    Autotune,
    /// GEMM (im2col) based implementation of convolution. Significantly increased memory usage.
    Gemm,
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
pub fn conv<R: CubeRuntime, const N: usize>(
    input: CubeTensor<R>,
    weight: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    options: ConvOptions<N>,
    strategy: ConvStrategy,
) -> Result<CubeTensor<R>, ConvSetupError> {
    let input = permute_nchw_to_nhwc(input);
    let weight = permute_nchw_to_nhwc(weight);

    let out = match strategy {
        ConvStrategy::Direct => conv_direct::<R, N>(input, weight, bias, options),
        #[cfg(feature = "autotune")]
        ConvStrategy::Autotune => Ok(conv_autotune::<R, N>(input, weight, bias, options)),
        ConvStrategy::Gemm => conv_im2col::<R, N>(input, weight, bias, options),
        ConvStrategy::ImplicitGemm => conv_gemm_cyclic::<R, N>(input, weight, bias, options),
    }?;

    Ok(permute_nhwc_to_nchw(out))
}
