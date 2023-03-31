use crate as burn;

use core::fmt::Display;

use crate::constant;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;

/// Applies the Gaussian Error Linear Units function element-wise.
#[derive(Clone, Debug, Default)]
pub struct GELU {}

constant!(GELU);

impl Display for GELU {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("GELU")
    }
}

impl GELU {
    /// Create the module.
    pub fn new() -> Self {
        Self {}
    }

    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - input: `[..., any]`
    /// - output: `[..., any]`
    pub fn forward<B: Backend, const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        crate::tensor::activation::gelu(input)
    }
}
