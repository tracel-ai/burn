use crate as burn;

use crate::module::Module;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;

/// Applies the Gaussian Error Linear Units function element-wise.
/// See also [gelu](burn::tensor::activation::gelu)
#[derive(Module, Clone, Debug, Default)]
pub struct Gelu {}

impl Gelu {
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
