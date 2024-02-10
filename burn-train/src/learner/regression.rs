use crate::metric::{Adaptor, ImageAccuracyInput, LossInput};
use burn_core::tensor::backend::Backend;
use burn_core::tensor::Tensor;

/// Simple regression output adapted for multiple metrics.
#[derive(new)]
pub struct RegressionOutput<B: Backend> {
    /// The loss.
    pub loss: Tensor<B, 1>,

    /// The output.
    pub output: Tensor<B, 2>,

    /// The targets.
    pub targets: Tensor<B, 2>,
}

impl<B: Backend> Adaptor<ImageAccuracyInput<B>> for RegressionOutput<B> {
    fn adapt(&self) -> ImageAccuracyInput<B> {
        ImageAccuracyInput::new(self.output.clone(), self.targets.clone())
    }
}

impl<B: Backend> Adaptor<LossInput<B>> for RegressionOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}
