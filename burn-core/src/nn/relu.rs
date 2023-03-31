use crate as burn;

use crate::constant;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;

/// Applies the rectified linear unit function element-wise:
///
/// `y = max(0, x)`
#[derive(Clone, Debug, Default)]
pub struct ReLU {}

constant!(ReLU);

impl ReLU {
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
        crate::tensor::activation::relu(input)
    }
}
