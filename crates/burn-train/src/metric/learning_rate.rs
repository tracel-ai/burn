use super::{
    state::{FormatOptions, NumericMetricState},
    MetricMetadata, Numeric,
};
use crate::metric::{Metric, MetricEntry};

/// Track the learning rate across iterations.
pub struct LearningRateMetric {
    state: NumericMetricState,
}

impl LearningRateMetric {
    /// Creates a new learning rate metric.
    pub fn new() -> Self {
        Self {
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

    fn name(&self) -> String {
        "Learning Rate".to_string()
    }
}

impl Numeric for LearningRateMetric {
    fn value(&self) -> f64 {
        self.state.value()
    }
}
