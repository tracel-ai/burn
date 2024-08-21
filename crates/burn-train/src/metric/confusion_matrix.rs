use burn_core::prelude::{Backend, Bool, Tensor};

#[derive(Clone, Debug)]
pub struct ConfusionMatrix<B: Backend> {
    pub true_positive: Tensor<B, 2, Bool>,
    pub false_positive: Tensor<B, 2, Bool>,
    pub true_negative: Tensor<B, 2, Bool>,
    pub false_negative: Tensor<B, 2, Bool>,
}

impl<B: Backend> ConfusionMatrix<B> {
    pub fn from(predictions: Tensor<B, 2, Bool>, targets: Tensor<B, 2, Bool>) -> Self {
        let stats_tensor = predictions.int() + targets.int() * 2;
        Self {
            true_positive: stats_tensor.clone().equal_elem(3),
            false_negative: stats_tensor.clone().equal_elem(2),
            false_positive: stats_tensor.clone().equal_elem(1),
            true_negative: stats_tensor.equal_elem(0)
        }
    }

    pub fn positive(self) -> Tensor<B, 2, Bool> {
        (self.true_positive.int() + self.false_negative.int()).bool()
    }

    pub fn negative(self) -> Tensor<B, 2, Bool> {
        self.positive().bool_not()
    }

    pub fn predicted_positive(self) -> Tensor<B, 2, Bool> {
        (self.true_positive.int() + self.false_positive.int()).bool()
    }

    pub fn predicted_negative(self) -> Tensor<B, 2, Bool> {
        self.predicted_positive().bool_not()
    }

}