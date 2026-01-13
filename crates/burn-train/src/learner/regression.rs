use crate::metric::processor::ItemLazy;
use crate::metric::{LossInput, LossMetric, MetricAdaptor};
use burn_core::tensor::backend::Backend;
use burn_core::tensor::{Tensor, Transaction};
use burn_ndarray::NdArray;

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

impl<B: Backend> MetricAdaptor<LossMetric<B>> for RegressionOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

impl<B: Backend> ItemLazy for RegressionOutput<B> {
    type ItemSync = RegressionOutput<NdArray>;

    fn sync(self) -> Self::ItemSync {
        let [output, loss, targets] = Transaction::default()
            .register(self.output)
            .register(self.loss)
            .register(self.targets)
            .execute()
            .try_into()
            .expect("Correct amount of tensor data");

        let device = &Default::default();

        RegressionOutput {
            output: Tensor::from_data(output, device),
            loss: Tensor::from_data(loss, device),
            targets: Tensor::from_data(targets, device),
        }
    }
}
