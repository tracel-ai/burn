use crate::metric::processor::ItemLazy;
use crate::metric::{Adaptor, LossInput, MaeInput, MseInput, R2ScoreInput};
use burn_core::tensor::Tensor;
use burn_std::ExecutionError;

/// Regression output adapted for regression metrics and loss.
///
/// Supported metrics:
/// - Loss
/// - MSE
/// - MAE
/// - R2 Score
#[derive(new)]
pub struct RegressionOutput {
    /// The loss.
    pub loss: Tensor<1>,

    /// The predicted values. Shape: \[batch_size, num_targets\].
    pub output: Tensor<2>,

    /// The ground truth values. Shape: \[batch_size, num_targets\].
    pub targets: Tensor<2>,
}

impl Adaptor<LossInput> for RegressionOutput {
    fn adapt(&self) -> LossInput {
        LossInput::new(self.loss.clone())
    }
}

impl Adaptor<MseInput> for RegressionOutput {
    fn adapt(&self) -> MseInput {
        MseInput::new(self.output.clone(), self.targets.clone())
    }
}

impl Adaptor<MaeInput> for RegressionOutput {
    fn adapt(&self) -> MaeInput {
        MaeInput::new(self.output.clone(), self.targets.clone())
    }
}

impl Adaptor<R2ScoreInput> for RegressionOutput {
    fn adapt(&self) -> R2ScoreInput {
        R2ScoreInput::new(self.output.clone(), self.targets.clone())
    }
}

impl ItemLazy for RegressionOutput {
    fn sync(self) -> Result<Self, ExecutionError> {
        // No readback: the metrics compute on the device the tensors live on
        // and read back only their final scalars. Flushing dispatches the
        // producing stream's buffered work so the metric thread doesn't wait
        // on an idle queue; a training item's float tensors come off the
        // autodiff backend entirely, so the metric thread neither retains the
        // tape nor pays its dispatch.
        self.loss.device().flush()?;

        Ok(RegressionOutput {
            output: self.output.without_autodiff(),
            loss: self.loss.without_autodiff(),
            targets: self.targets.without_autodiff(),
        })
    }
}
