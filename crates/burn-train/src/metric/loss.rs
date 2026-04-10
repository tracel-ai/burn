use std::sync::Arc;

use super::MetricMetadata;
use super::SerializedEntry;
use super::state::FormatOptions;
use super::state::NumericMetricState;
use crate::metric::MetricName;
use crate::metric::{Metric, MetricAttributes, Numeric, NumericAttributes, NumericEntry};
use burn_core::tensor::Tensor;

/// The loss metric.
#[derive(Clone)]
pub struct LossMetric {
    name: Arc<String>,
    state: NumericMetricState,
}

/// The [loss metric](LossMetric) input type.
#[derive(new)]
pub struct LossInput {
    tensor: Tensor<1>,
}

impl Default for LossMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl LossMetric {
    /// Create the metric.
    pub fn new() -> Self {
        Self {
            name: Arc::new("Loss".to_string()),
            state: NumericMetricState::default(),
        }
    }
}

impl Metric for LossMetric {
    type Input = LossInput;

    fn update(&mut self, loss: &Self::Input, _metadata: &MetricMetadata) -> SerializedEntry {
        let [batch_size] = loss.tensor.dims();
        let loss = loss
            .tensor
            .clone()
            .mean()
            .into_data()
            .iter::<f64>()
            .next()
            .unwrap();

        self.state.update(
            loss,
            batch_size,
            FormatOptions::new(self.name()).precision(2),
        )
    }

    fn clear(&mut self) {
        self.state.reset()
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

impl Numeric for LossMetric {
    fn value(&self) -> NumericEntry {
        self.state.current_value()
    }

    fn running_value(&self) -> NumericEntry {
        self.state.running_value()
    }
}
