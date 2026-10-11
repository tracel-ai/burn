use super::MetricMetadata;
use super::state::{FormatOptions, NumericMetricState};
use crate::metric::{Metric, MetricAttributes, MetricName, Numeric, SerializedEntry};
use burn_core::tensor::{Tensor, TensorReadError};

/// The Mean Absolute Error (MAE) metric for regression tasks.
#[derive(Clone)]
pub struct MAEMetric {
    name: MetricName,
    state: NumericMetricState,
}

/// The [MAE metric](MAEMetric) input type.
#[derive(new)]
pub struct MAEInput {
    /// The model outputs.
    pub outputs: Tensor<2>,
    /// The targets.
    pub targets: Tensor<2>,
}

/// Rust-style alias for [`MAEMetric`].
pub type MaeMetric = MAEMetric;

impl Default for MAEMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl MAEMetric {
    /// Creates the metric.
    pub fn new() -> Self {
        Self {
            name: MetricName::new("MAE".to_string()),
            state: Default::default(),
        }
    }
}

impl Metric for MAEMetric {
    type Input = MAEInput;

    fn update(
        &mut self,
        input: &MAEInput,
        _metadata: &MetricMetadata,
    ) -> Result<SerializedEntry, TensorReadError> {
        let targets = input.targets.clone();
        let outputs = input.outputs.clone();

        assert_eq!(outputs.dims(), targets.dims(), "MAE shapes must match");
        let [batch_size, num_targets] = outputs.dims();
        assert!(
            batch_size > 0 && num_targets > 0,
            "MAE requires non-empty input"
        );

        let diff = outputs.sub(targets);
        let mae = diff.abs().mean();

        self.state
            .update(mae.try_into_scalar::<f64>()?, batch_size * num_targets);
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
        self.state.reset()
    }

    fn name(&self) -> MetricName {
        self.name.clone()
    }

    fn attributes(&self) -> MetricAttributes {
        super::NumericAttributes {
            unit: None,
            higher_is_better: false,
        }
        .into()
    }
}

impl Numeric for MAEMetric {
    fn value(&self) -> Option<super::NumericEntry> {
        Some(self.state.current_value())
    }

    fn running_value(&self) -> Option<super::NumericEntry> {
        Some(self.state.running_value())
    }

    fn final_value(&self) -> super::NumericEntry {
        self.state.final_value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sklearn_reference_and_uneven_batches() {
        // scikit-learn 1.8.0: mean_absolute_error(y_true, y_pred) == 0.85.
        let device = Default::default();
        let targets = [[1., 10.], [2., 10.], [3., 12.], [4., 14.], [5., 18.]];
        let outputs = [[1., 9.], [2.5, 11.], [2., 11.], [5., 14.], [4., 20.]];
        let mut metric = MaeMetric::new();
        metric
            .update(
                &MAEInput::new(
                    Tensor::from_data(outputs, &device),
                    Tensor::from_data(targets, &device),
                ),
                &MetricMetadata::fake(),
            )
            .unwrap();
        assert!((metric.final_value().current() - 0.85).abs() < 1e-6);

        metric.clear();
        for range in [0..2, 2..5] {
            metric
                .update(
                    &MAEInput::new(
                        Tensor::from_data(outputs, &device).slice([range.clone()]),
                        Tensor::from_data(targets, &device).slice([range]),
                    ),
                    &MetricMetadata::fake(),
                )
                .unwrap();
        }
        assert!((metric.running_value().unwrap().current() - 0.85).abs() < 1e-6);
        metric.compute().unwrap();
        assert!((metric.final_value().current() - 0.85).abs() < 1e-6);
    }
}
