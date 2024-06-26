use crate as burn;

use crate::nn::loss::reduction::Reduction;

use crate::module::Module;
use crate::tensor::{backend::Backend, Tensor};

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
        logits.sub(targets).powf_scalar(2.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Data;
    use crate::TestBackend;

    #[test]
    fn test_mse_loss() {
        let device = Default::default();
        let logits =
            Tensor::<TestBackend, 2>::from_data(Data::from([[1.0, 2.0], [3.0, 4.0]]), &device);

        let targets =
            Tensor::<TestBackend, 2>::from_data(Data::from([[2.0, 1.0], [3.0, 2.0]]), &device);

        let mse = MseLoss::new();
        let loss_no_reduction = mse.forward_no_reduction(logits.clone(), targets.clone());
        let loss = mse.forward(logits.clone(), targets.clone(), Reduction::Auto);
        let loss_sum = mse.forward(logits, targets, Reduction::Sum);

        assert_eq!(
            loss_no_reduction.into_data(),
            Data::from([[1.0, 1.0], [0.0, 4.0]])
        );
        assert_eq!(loss.into_data(), Data::from([1.5]));
        assert_eq!(loss_sum.into_data(), Data::from([6.0]));
    }

    #[test]
    fn display() {
        let loss = MseLoss::new();
        assert_eq!(alloc::format!("{}", loss), "MseLoss");
    }
}
