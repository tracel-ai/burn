use burn_core as burn;

use crate::loss::reduction::Reduction;

use burn::module::Module;
use burn::tensor::Tensor;
use burn::tensor::activation::softplus;

/// Calculate the soft margin loss between the input logits and the target tensor.
///
/// The target values are expected to be `-1` or `1`. The loss for each element is
/// `log(1 + exp(-target * logit))`, matching `torch.nn.SoftMarginLoss`.
#[derive(Module, Debug)]
pub struct SoftMarginLoss;

impl Default for SoftMarginLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl SoftMarginLoss {
    /// Create the criterion.
    pub fn new() -> Self {
        Self
    }

    /// Compute the criterion on the input tensor.
    ///
    /// # Shapes
    ///
    /// - logits: `[batch_size, num_targets]`
    /// - targets: `[batch_size, num_targets]` (values in `{-1, 1}`)
    pub fn forward<const D: usize>(
        &self,
        logits: Tensor<D>,
        targets: Tensor<D>,
        reduction: Reduction,
    ) -> Tensor<1> {
        let tensor = self.forward_no_reduction(logits, targets);
        match reduction {
            Reduction::Mean | Reduction::Auto => tensor.mean(),
            Reduction::Sum => tensor.sum(),
            other => panic!("{other:?} reduction is not supported"),
        }
    }

    /// Compute the criterion on the input tensor without reducing.
    pub fn forward_no_reduction<const D: usize>(
        &self,
        logits: Tensor<D>,
        targets: Tensor<D>,
    ) -> Tensor<D> {
        // log(1 + exp(-target * logit)) = softplus(-target * logit)
        softplus(targets.mul(logits).neg(), 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use burn::tensor::Tolerance;
    type FT = f32;

    #[test]
    fn test_soft_margin_loss() {
        let device = Default::default();
        let logits = Tensor::<2>::from_data(TensorData::from([[0.5, -1.0], [2.0, -0.5]]), &device);
        let targets =
            Tensor::<2>::from_data(TensorData::from([[1.0, -1.0], [-1.0, 1.0]]), &device);

        let loss = SoftMarginLoss::new();
        let no_reduction = loss.forward_no_reduction(logits.clone(), targets.clone());
        let mean = loss.forward(logits.clone(), targets.clone(), Reduction::Mean);
        let sum = loss.forward(logits, targets, Reduction::Sum);

        // log(1 + exp(-target * logit)), computed with an independent reference.
        let expected = TensorData::from([[0.474077, 0.313262], [2.126928, 0.974077]]);
        no_reduction
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());

        mean.into_data()
            .assert_approx_eq::<FT>(&TensorData::from([0.972086]), Tolerance::default());
        sum.into_data()
            .assert_approx_eq::<FT>(&TensorData::from([3.888344]), Tolerance::default());
    }
}
