use super::MetricMetadata;
use super::state::{FormatOptions, NumericMetricState};
use crate::metric::{Metric, MetricAttributes, MetricName, Numeric, SerializedEntry};
use burn_core::tensor::Tensor;

/// The Mean Squared Error (MSE) metric for regression tasks.
#[derive(Clone)]
pub struct MSEMetric {
    name: MetricName,
    state: NumericMetricState,
}

/// The [MSE metric](MSEMetric) input type.
#[derive(new)]
pub struct MSEInput {
    /// The model outputs.
    pub outputs: Tensor<2>,
    /// The targets.
    pub targets: Tensor<2>,
}

impl Default for MSEMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl MSEMetric {
    /// Creates the metric.
    pub fn new() -> Self {
        Self {
            name: MetricName::new("MSE".to_string()),
            state: Default::default(),
        }
    }
}

impl Metric for MSEMetric {
    type Input = MSEInput;

    fn update(&mut self, input: &MSEInput, _metadata: &MetricMetadata) -> SerializedEntry {
        let targets = input.targets.clone();
        let outputs = input.outputs.clone();

        let [batch_size, _] = outputs.dims();

        let diff = outputs.sub(targets);
        let mse = diff.powf_scalar(2.0).mean();

        self.state.update(
            mse.into_scalar::<f64>(),
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

impl Numeric for MSEMetric {
    fn value(&self) -> super::NumericEntry {
        self.state.current_value()
    }

    fn running_value(&self) -> super::NumericEntry {
        self.state.running_value()
    }
}
