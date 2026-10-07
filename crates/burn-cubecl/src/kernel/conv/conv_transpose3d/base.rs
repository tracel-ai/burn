use crate::tensor::CubeTensor;
use burn_backend::ops::ConvTransposeOptions;
use cubek::convolution::components::ConvSetupError;

#[cfg(feature = "autotune")]
use super::conv_transpose3d_autotune;
use super::{conv_transpose3d_col2im, conv_transpose3d_direct};

/// The strategy to be used when launching a 3D conv_transpose kernel.
pub enum ConvTranspose3dStrategy {
    /// A simple direct convolution.
    Direct,
    #[cfg(feature = "autotune")]
    /// Using autotune to choose the best kernel based on runtime information.
    Autotune,
    /// GEMM (col2im) based implementation of convolution. Significantly increased memory usage.
    Gemm,
}

impl Default for ConvTranspose3dStrategy {
    fn default() -> Self {
        // if autotune is enabled, default to autotune
        #[cfg(feature = "autotune")]
        return ConvTranspose3dStrategy::Autotune;

        // if autotune is disabled, default to the more memory-conservative algorithm
        #[cfg(not(feature = "autotune"))]
        ConvTranspose3dStrategy::Direct
    }
}

/// Performs a 3D transposed convolution with the given strategy
///
/// * `input` - The input feature map
/// * `weight` - The weights (filter) applied to each kernel
/// * `bias` - The bias added to each channel
/// * `options` - The options to use for the convolution
/// * `strategy` - The convolution algorithm to use. Autotune will pick the fastest available option.
pub fn conv_transpose3d(
    input: CubeTensor,
    weight: CubeTensor,
    bias: Option<CubeTensor>,
    options: ConvTransposeOptions<3>,
    strategy: ConvTranspose3dStrategy,
) -> Result<CubeTensor, ConvSetupError> {
    match strategy {
        ConvTranspose3dStrategy::Direct => conv_transpose3d_direct(input, weight, bias, options),
        #[cfg(feature = "autotune")]
        ConvTranspose3dStrategy::Autotune => {
            Ok(conv_transpose3d_autotune(input, weight, bias, options))
        }
        ConvTranspose3dStrategy::Gemm => conv_transpose3d_col2im(input, weight, bias, options),
    }
}
