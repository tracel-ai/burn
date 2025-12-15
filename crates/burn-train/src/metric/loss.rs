use std::sync::Arc;

use super::MetricMetadata;
use super::SerializedEntry;
use super::state::FormatOptions;
use super::state::NumericMetricState;
use crate::metric::MetricName;
use crate::metric::{Metric, MetricAttributes, Numeric, NumericAttributes, NumericEntry};
use burn_core::tensor::Tensor;
use burn_core::tensor::backend::Backend;

/// The loss metric.
#[derive(Clone)]
pub struct LossMetric<B: Backend> {
    name: Arc<String>,
    state: NumericMetricState,
    _b: B,
}

/// The [loss metric](LossMetric) input type.
#[derive(new)]
pub struct LossInput<B: Backend> {
    tensor: Tensor<B, 1>,
}

impl<B: Backend> Default for LossMetric<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> LossMetric<B> {
    /// Create the metric.
    pub fn new() -> Self {
        Self {
            name: Arc::new("Loss".to_string()),
            state: NumericMetricState::default(),
            _b: Default::default(),
        }
    }
}

impl<B: Backend> Metric for LossMetric<B> {
    type Input = LossInput<B>;

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

impl<B: Backend> Numeric for LossMetric<B> {
    fn value(&self) -> NumericEntry {
        self.state.current_value()
    }

    fn running_value(&self) -> super::NumericEntry {
        self.state.running_value()
    }
}
