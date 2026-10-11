use crate::metric::processor::ItemLazy;
use crate::metric::{Adaptor, LossInput, MAEInput, MSEInput, RegressionInput};
use burn_core::tensor::Tensor;
use burn_std::ExecutionError;

/// Regression output adapted for loss, MAE, MSE, and R² metrics.
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

impl Adaptor<MAEInput> for RegressionOutput {
    fn adapt(&self) -> MAEInput {
        MAEInput::new(self.output.clone(), self.targets.clone())
    }
}

impl Adaptor<MSEInput> for RegressionOutput {
    fn adapt(&self) -> MSEInput {
        MSEInput::new(self.output.clone(), self.targets.clone())
    }
}

impl Adaptor<RegressionInput> for RegressionOutput {
    fn adapt(&self) -> RegressionInput {
        RegressionInput::new(self.output.clone(), self.targets.clone())
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
