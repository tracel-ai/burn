use super::MetricMetadata;
use super::state::{FormatOptions, NumericMetricState};
use crate::metric::{Metric, MetricAttributes, MetricName, Numeric, SerializedEntry};
use burn_core::tensor::Tensor;

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

    fn update(&mut self, input: &MAEInput, _metadata: &MetricMetadata) -> SerializedEntry {
        let targets = input.targets.clone();
        let outputs = input.outputs.clone();

        let [batch_size, _] = outputs.dims();

        let diff = outputs.sub(targets);
        let mae = diff.abs().mean();

        self.state.update(
            mae.into_scalar::<f64>(),
            batch_size,
            FormatOptions::new(self.name()).precision(4),
        )
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
            ..Default::default()
        }
        .into()
    }
}

impl Numeric for MAEMetric {
    fn value(&self) -> super::NumericEntry {
        self.state.current_value()
    }

    fn running_value(&self) -> super::NumericEntry {
        self.state.running_value()
    }
}
