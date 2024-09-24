use burn_tensor::{
    ops::{ConvOptions, ConvTransposeOptions},
    TensorData,
};

use crate::{tensor::JitTensor, FloatElement, IntElement, JitElement, JitRuntime};

#[cfg(feature = "autotune")]
use super::conv2d_autotune;
use super::{
    conv2d_direct, conv2d_im2col, conv_transpose2d_autotune, conv_transpose2d_col2im,
    conv_transpose2d_direct, implicit_gemm::conv2d_implicit_gemm,
};

/// The strategy to be used when launching a convolution kernel.
pub enum Conv2dStrategy {
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

impl Default for Conv2dStrategy {
    fn default() -> Self {
        // if autotune is enabled, default to autotune
        #[cfg(feature = "autotune")]
        return Conv2dStrategy::Autotune;

        // if autotune is disabled, default to the more memory-conservative algorithm
        #[cfg(not(feature = "autotune"))]
        Conv2dStrategy::Direct
    }
}

/// The strategy to be used when launching a conv_transpose kernel.
pub enum ConvTranspose2dStrategy {
    /// A simple direct convolution.
    Direct,
    #[cfg(feature = "autotune")]
    /// Using autotune to choose the best kernel based on runtime information.
    Autotune,
    /// GEMM (im2col) based implementation of convolution. Significantly increased memory usage.
    Gemm,
}

impl Default for ConvTranspose2dStrategy {
    fn default() -> Self {
        // if autotune is enabled, default to autotune
        #[cfg(feature = "autotune")]
        return ConvTranspose2dStrategy::Autotune;

        // if autotune is disabled, default to the more memory-conservative algorithm
        #[cfg(not(feature = "autotune"))]
        ConvTranspose2dStrategy::Direct
    }
}

/// Perform a 2D convolution with the given strategy
///
/// * `input` - The input feature map
/// * `weight` - The weights (filter) applied to each kernel
/// * `bias` - The bias added to each channel
/// * `options` - The options to use for the convolution
/// * `strategy` - The convolution algorithm to use. Autotune will pick the fastest available option.
///
pub fn conv2d<R: JitRuntime, E: FloatElement, I: IntElement>(
    input: JitTensor<R, E>,
    weight: JitTensor<R, E>,
    bias: Option<JitTensor<R, E>>,
    options: ConvOptions<2>,
    strategy: Conv2dStrategy,
) -> JitTensor<R, E> {
    match strategy {
        Conv2dStrategy::Direct => conv2d_direct::<R, E, I>(input, weight, bias, options),
        #[cfg(feature = "autotune")]
        Conv2dStrategy::Autotune => conv2d_autotune::<R, E, I>(input, weight, bias, options),
        Conv2dStrategy::Gemm => conv2d_im2col::<R, E, I>(input, weight, bias, options),
        Conv2dStrategy::ImplicitGemm => {
            conv2d_implicit_gemm::<R, E, I>(input, weight, bias, options)
        }
    }
}

/// Perform a 2D convolution with the given strategy
///
/// * `input` - The input feature map
/// * `weight` - The weights (filter) applied to each kernel
/// * `bias` - The bias added to each channel
/// * `options` - The options to use for the convolution
/// * `strategy` - The convolution algorithm to use. Autotune will pick the fastest available option.
///
pub fn conv_transpose2d<R: JitRuntime, E: FloatElement, I: IntElement>(
    input: JitTensor<R, E>,
    weight: JitTensor<R, E>,
    bias: Option<JitTensor<R, E>>,
    options: ConvTransposeOptions<2>,
    strategy: ConvTranspose2dStrategy,
) -> JitTensor<R, E> {
    match strategy {
        ConvTranspose2dStrategy::Direct => {
            conv_transpose2d_direct::<R, E, I>(input, weight, bias, options)
        }
        #[cfg(feature = "autotune")]
        ConvTranspose2dStrategy::Autotune => {
            conv_transpose2d_autotune::<R, E, I>(input, weight, bias, options)
        }
        ConvTranspose2dStrategy::Gemm => {
            conv_transpose2d_col2im::<R, E, I>(input, weight, bias, options)
        }
    }
}

#[allow(unused)]
pub(crate) fn debug_data<R: JitRuntime, E: JitElement>(tensor: JitTensor<R, E>) -> TensorData {
    let bytes = tensor.client.read(tensor.handle.binding());
    TensorData::new(E::from_bytes(&bytes).to_vec(), tensor.shape)
}
