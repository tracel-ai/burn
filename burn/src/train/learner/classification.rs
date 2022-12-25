use crate::tensor::backend::Backend;
use crate::train::metric::{AccuracyInput, Adaptor};
use burn_tensor::Tensor;

#[derive(new)]
pub struct ClassificationOutput<B: Backend> {
    pub loss: Tensor<B, 1>,
    pub output: Tensor<B, 2>,
    pub targets: Tensor<B::IntegerBackend, 1>,
}

impl<B: Backend> Adaptor<AccuracyInput<B>> for ClassificationOutput<B> {
    fn adapt(&self) -> AccuracyInput<B> {
        AccuracyInput::new(self.output.clone(), self.targets.clone())
    }
}

impl<B: Backend> Adaptor<Tensor<B, 1>> for ClassificationOutput<B> {
    fn adapt(&self) -> Tensor<B, 1> {
        self.loss.clone()
    }
}
