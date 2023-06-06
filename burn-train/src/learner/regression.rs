use crate::metric::{Adaptor, LossInput};
use burn_core::tensor::backend::Backend;
use burn_core::tensor::Tensor;

/// Simple regression output adapted for multiple metrics.
#[derive(new)]
pub struct RegressionOutput<B: Backend> {
    pub loss: Tensor<B, 1>,
    pub output: Tensor<B, 2>,
    pub targets: Tensor<B, 2>,
}

impl<B: Backend> Adaptor<LossInput<B>> for RegressionOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}
