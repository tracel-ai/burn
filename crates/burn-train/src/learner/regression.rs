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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::{
        MaeMetric, Metric, MetricAttributes, MetricMetadata, MseMetric, Numeric, NumericEntry,
        R2ScoreMetric,
    };

    fn assert_metadata(metric: &impl Metric, name: &str, higher_is_better: bool) {
        assert_eq!(metric.name().as_str(), name);
        let MetricAttributes::Numeric(attributes) = metric.attributes() else {
            panic!("Regression metrics must expose numeric attributes");
        };
        assert_eq!(attributes.unit, None);
        assert_eq!(attributes.higher_is_better, higher_is_better);
    }

    fn assert_batch(metric: &impl Numeric, expected: f64, count: usize) {
        let NumericEntry::Aggregated {
            aggregated_value,
            count: actual_count,
        } = metric.value().unwrap()
        else {
            panic!("Regression metrics must expose their current batch value");
        };
        assert!((aggregated_value - expected).abs() < 1e-6);
        assert_eq!(actual_count, count);
    }

    fn assert_epoch(metric: &mut (impl Metric + Numeric), expected: f64) {
        assert!((metric.running_value().unwrap().current() - expected).abs() < 1e-6);
        let entry = metric.compute().unwrap();
        let NumericEntry::Final(value) = NumericEntry::deserialize(&entry.serialized).unwrap()
        else {
            panic!("The serialized epoch result must be marked final");
        };
        assert!((value - expected).abs() < 1e-6);
        assert!((metric.final_value().current() - expected).abs() < 1e-6);
    }

    #[test]
    fn regression_output_drives_metrics_across_batches_and_epochs() {
        let device = Default::default();
        let output = RegressionOutput::new(
            Tensor::from_data([1.0], &device),
            Tensor::from_data([[0.0], [2.0]], &device),
            Tensor::from_data([[1.0], [3.0]], &device),
        )
        .sync()
        .unwrap();
        let mut mae = MaeMetric::default();
        let mut mse = MseMetric::default();
        let mut r2 = R2ScoreMetric::default();
        assert_metadata(&mae, "MAE", false);
        assert_metadata(&mse, "MSE", false);
        assert_metadata(&r2, "R2 Score", true);

        mae.update(&output.adapt(), &MetricMetadata::fake())
            .unwrap();
        mse.update(&output.adapt(), &MetricMetadata::fake())
            .unwrap();
        r2.update(&output.adapt(), &MetricMetadata::fake()).unwrap();
        assert_batch(&mae, 1.0, 2);
        assert_batch(&mse, 1.0, 2);
        assert_batch(&r2, 0.0, 2);

        let output = RegressionOutput::new(
            Tensor::from_data([2.0], &device),
            Tensor::from_data([[4.0], [8.0], [7.0]], &device),
            Tensor::from_data([[5.0], [7.0], [9.0]], &device),
        )
        .sync()
        .unwrap();
        mae.update(&output.adapt(), &MetricMetadata::fake())
            .unwrap();
        mse.update(&output.adapt(), &MetricMetadata::fake())
            .unwrap();
        r2.update(&output.adapt(), &MetricMetadata::fake()).unwrap();
        assert_batch(&mae, 4.0 / 3.0, 3);
        assert_batch(&mse, 2.0, 3);
        assert_batch(&r2, 0.25, 3);
        assert_epoch(&mut mae, 1.2);
        assert_epoch(&mut mse, 1.6);
        assert_epoch(&mut r2, 0.8);

        mae.clear();
        mse.clear();
        r2.clear();
        assert!(r2.value().is_none());
        assert!(r2.running_value().is_none());
        mae.update(&output.adapt(), &MetricMetadata::fake())
            .unwrap();
        mse.update(&output.adapt(), &MetricMetadata::fake())
            .unwrap();
        r2.update(&output.adapt(), &MetricMetadata::fake()).unwrap();
        assert_epoch(&mut mae, 4.0 / 3.0);
        assert_epoch(&mut mse, 2.0);
        assert_epoch(&mut r2, 0.25);
    }
}
