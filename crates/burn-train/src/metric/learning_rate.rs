use std::sync::Arc;

use super::{
    MetricMetadata, Numeric,
    state::{FormatOptions, NumericMetricState},
};
use crate::metric::{Metric, MetricEntry, MetricName};

/// Track the learning rate across iterations.
#[derive(Clone)]
pub struct LearningRateMetric {
    name: MetricName,
    state: NumericMetricState,
}

impl LearningRateMetric {
    /// Creates a new learning rate metric.
    pub fn new() -> Self {
        Self {
            name: Arc::new("Learning Rate".to_string()),
            state: NumericMetricState::new(),
        }
    }
}

impl Default for LearningRateMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl Metric for LearningRateMetric {
    type Input = ();

    fn update(&mut self, _item: &(), metadata: &MetricMetadata) -> MetricEntry {
        let lr = metadata.lr.unwrap_or(0.0);

        self.state
            .update(lr, 1, FormatOptions::new(self.name()).precision(2))
    }

    fn clear(&mut self) {
        self.state.reset()
    }

    fn name(&self) -> MetricName {
        self.name.clone()
    }
}

impl Numeric for LearningRateMetric {
    fn value(&self) -> super::NumericEntry {
        self.state.value()
    }
}
