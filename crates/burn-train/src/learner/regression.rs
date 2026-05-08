use crate::metric::processor::ItemLazy;
use crate::metric::{Adaptor, LossInput};
use burn_core::tensor::{Device, Tensor, Transaction};
use burn_flex::FlexDevice;

/// Regression output adapted for the loss metric.
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

impl ItemLazy for RegressionOutput {
    fn sync(self) -> Self {
        let [output, loss, targets] = Transaction::default()
            .register(self.output)
            .register(self.loss)
            .register(self.targets)
            .execute()
            .try_into()
            .expect("Correct amount of tensor data");

        let device: Device = FlexDevice.into();

        RegressionOutput {
            output: Tensor::from_data(output, &device),
            loss: Tensor::from_data(loss, &device),
            targets: Tensor::from_data(targets, &device),
        }
    }
}
