use std::sync::Arc;

use super::MetricMetadata;
use super::state::{FormatOptions, NumericMetricState};
use crate::metric::{
    Metric, MetricAttributes, MetricName, Numeric, NumericAttributes, NumericEntry, SerializedEntry,
};
use burn_core::tensor::{Tensor, TensorReadError};

/// The Mean Absolute Error (MAE) metric for regression tasks.
#[derive(Clone)]
pub struct MaeMetric {
    name: MetricName,
    state: NumericMetricState,
}

/// The [MAE metric](MaeMetric) input type.
#[derive(new)]
pub struct MaeInput {
    /// The model outputs.
    pub outputs: Tensor<2>,
    /// The targets.
    pub targets: Tensor<2>,
}

/// Type alias for [`MaeMetric`].
pub type MAEMetric = MaeMetric;
/// Type alias for [`MaeInput`].
pub type MAEInput = MaeInput;

impl Default for MaeMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl MaeMetric {
    /// Creates the metric.
    pub fn new() -> Self {
        Self {
            name: Arc::new("MAE".to_string()),
            state: NumericMetricState::default(),
        }
    }
}

impl Metric for MaeMetric {
    type Input = MaeInput;

    fn update(
        &mut self,
        input: &Self::Input,
        _metadata: &MetricMetadata,
    ) -> Result<SerializedEntry, TensorReadError> {
        assert_eq!(
            input.outputs.dims(),
            input.targets.dims(),
            "MAE outputs and targets must have the same shape"
        );
        let [batch_size, _] = input.outputs.dims();
        let diff = input.outputs.clone().sub(input.targets.clone());
        let mae = diff.abs().mean().try_into_scalar::<f64>()?;

        self.state.update(mae, batch_size);
        Ok(self
            .state
            .compute_update(FormatOptions::new(self.name()).precision(4)))
    }

    fn compute(&mut self) -> Result<SerializedEntry, TensorReadError> {
        Ok(self
            .state
            .compute_final(FormatOptions::new(self.name()).precision(4)))
    }

    fn clear(&mut self) {
        self.state.reset();
    }

    fn name(&self) -> MetricName {
        self.name.clone()
    }

    fn attributes(&self) -> MetricAttributes {
        NumericAttributes {
            unit: None,
            higher_is_better: false,
        }
        .into()
    }
}

impl Numeric for MaeMetric {
    fn value(&self) -> Option<NumericEntry> {
        Some(self.state.current_value())
    }

    fn running_value(&self) -> Option<NumericEntry> {
        Some(self.state.running_value())
    }

    fn final_value(&self) -> NumericEntry {
        self.state.final_value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learner::RegressionOutput;
    use crate::metric::Adaptor;
    use burn_core::tensor::Device;

    #[test]
    fn test_mae_perfect_prediction() {
        let device = Device::default();
        let mut metric = MaeMetric::new();

        let input = MaeInput::new(
            Tensor::from_data([[1.0, 2.0], [3.0, 4.0]], &device),
            Tensor::from_data([[1.0, 2.0], [3.0, 4.0]], &device),
        );

        let entry = metric.update(&input, &MetricMetadata::fake()).unwrap();
        assert_eq!(metric.final_value().current(), 0.0);
        assert_eq!(entry.formatted, "epoch 0.0000e0 - batch 0.0000e0");
    }

    #[test]
    fn test_mae_known_values() {
        let device = Device::default();
        let mut metric = MaeMetric::new();

        // Predictions: [[1.0, 2.0], [3.0, 4.0]]
        // Targets:     [[1.5, 0.5], [2.0, 6.0]]
        // Residuals:   [-0.5, 1.5, 1.0, -2.0]
        // Absolute:    [0.5, 1.5, 1.0, 2.0] -> sum = 5.0 -> mean = 1.25
        let input = MaeInput::new(
            Tensor::from_data([[1.0, 2.0], [3.0, 4.0]], &device),
            Tensor::from_data([[1.5, 0.5], [2.0, 6.0]], &device),
        );

        let _ = metric.update(&input, &MetricMetadata::fake()).unwrap();
        let current = metric.value().unwrap().current();
        assert!((current - 1.25).abs() < 1e-6);
        assert!((metric.final_value().current() - 1.25).abs() < 1e-6);
    }

    #[test]
    fn test_mae_multiple_batches() {
        let device = Device::default();
        let mut metric = MaeMetric::new();

        // Batch 1: batch_size = 2, error = 1.0
        let input1 = MaeInput::new(
            Tensor::from_data([[1.0], [2.0]], &device),
            Tensor::from_data([[2.0], [3.0]], &device),
        );
        let _ = metric.update(&input1, &MetricMetadata::fake()).unwrap();

        // Batch 2: batch_size = 4, error = 2.0 (diff = 2.0, abs = 2.0)
        let input2 = MaeInput::new(
            Tensor::from_data([[0.0], [0.0], [0.0], [0.0]], &device),
            Tensor::from_data([[2.0], [2.0], [2.0], [2.0]], &device),
        );
        let _ = metric.update(&input2, &MetricMetadata::fake()).unwrap();

        // Weighted running average: (2 * 1.0 + 4 * 2.0) / 6 = 10.0 / 6 = 1.666666...
        let expected = 10.0 / 6.0;
        assert!((metric.final_value().current() - expected).abs() < 1e-6);
    }

    #[test]
    fn test_mae_with_regression_output() {
        let device = Device::default();
        let mut metric = MaeMetric::new();

        let output = RegressionOutput::new(
            Tensor::from_data([0.5], &device),
            Tensor::from_data([[1.0, 2.0]], &device),
            Tensor::from_data([[2.0, 0.0]], &device),
        );

        let input: MaeInput = output.adapt();
        let _ = metric.update(&input, &MetricMetadata::fake()).unwrap();

        // diffs: [-1.0, 2.0] -> abs: [1.0, 2.0] -> mean: 1.5
        assert!((metric.final_value().current() - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_mae_shape_mismatch_panics() {
        let device = Device::default();
        let mut metric = MaeMetric::new();

        let input = MaeInput::new(
            Tensor::from_data([[1.0, 2.0]], &device),
            Tensor::from_data([[1.0]], &device),
        );

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _ = metric.update(&input, &MetricMetadata::fake());
        }));
        assert!(result.is_err());
    }

    #[test]
    fn test_mae_clear() {
        let device = Device::default();
        let mut metric = MaeMetric::new();

        let input = MaeInput::new(
            Tensor::from_data([[1.0]], &device),
            Tensor::from_data([[3.0]], &device),
        );
        let _ = metric.update(&input, &MetricMetadata::fake()).unwrap();
        assert!(metric.final_value().current() > 0.0);

        metric.clear();
        assert!(metric.final_value().current().is_nan());
    }
}
