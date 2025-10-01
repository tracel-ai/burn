use burn_core as burn;

use crate::loss::reduction::Reduction;

use burn::module::Module;
use burn::tensor::{Tensor, backend::Backend};

/// Calculate the mean squared error loss from the input logits and the targets.
#[derive(Module, Clone, Debug)]
pub struct MseLoss;

impl Default for MseLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl MseLoss {
    /// Create the criterion.
    pub fn new() -> Self {
        Self
    }

    /// Compute the criterion on the input tensor.
    ///
    /// # Shapes
    ///
    /// - logits: [batch_size, num_targets]
    /// - targets: [batch_size, num_targets]
    pub fn forward<const D: usize, B: Backend>(
        &self,
        logits: Tensor<B, D>,
        targets: Tensor<B, D>,
        reduction: Reduction,
    ) -> Tensor<B, 1> {
        let tensor = self.forward_no_reduction(logits, targets);
        match reduction {
            Reduction::Mean | Reduction::Auto => tensor.mean(),
            Reduction::Sum => tensor.sum(),
        }
    }

    /// Compute the criterion on the input tensor without reducing.
    pub fn forward_no_reduction<const D: usize, B: Backend>(
        &self,
        logits: Tensor<B, D>,
        targets: Tensor<B, D>,
    ) -> Tensor<B, D> {
        logits.sub(targets).powi_scalar(2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn::tensor::TensorData;

    #[test]
    fn test_mse_loss() {
        let device = Default::default();
        let logits = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0, 2.0], [3.0, 4.0]]),
            &device,
        );

        let targets = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[2.0, 1.0], [3.0, 2.0]]),
            &device,
        );

        let mse = MseLoss::new();
        let loss_no_reduction = mse.forward_no_reduction(logits.clone(), targets.clone());
        let loss = mse.forward(logits.clone(), targets.clone(), Reduction::Auto);
        let loss_sum = mse.forward(logits, targets, Reduction::Sum);

        let expected = TensorData::from([[1.0, 1.0], [0.0, 4.0]]);
        loss_no_reduction.into_data().assert_eq(&expected, false);

        let expected = TensorData::from([1.5]);
        loss.into_data().assert_eq(&expected, false);

        let expected = TensorData::from([6.0]);
        loss_sum.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn display() {
        let loss = MseLoss::new();
        assert_eq!(alloc::format!("{loss}"), "MseLoss");
    }
}
