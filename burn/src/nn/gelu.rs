use crate::module::Forward;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;

/// Applies the Gaussian Error Linear Units function element-wise.
#[derive(Clone, Debug, Default)]
pub struct GELU {}

impl GELU {
    pub fn new() -> Self {
        Self {}
    }
}

impl<B: Backend, const D: usize> Forward<Tensor<B, D>, Tensor<B, D>> for GELU {
    fn forward(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        crate::tensor::activation::gelu(&input)
    }
}
