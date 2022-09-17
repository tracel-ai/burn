use crate::module::Forward;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;

/// Applies the rectified linear unit function element-wise:
///
/// `y = max(0, x)`
#[derive(Clone, Debug, Default)]
pub struct ReLU {}

impl ReLU {
    pub fn new() -> Self {
        Self {}
    }
}

impl<B: Backend, const D: usize> Forward<Tensor<B, D>, Tensor<B, D>> for ReLU {
    fn forward(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        crate::tensor::activation::relu(&input)
    }
}
