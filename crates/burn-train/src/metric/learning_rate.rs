use std::sync::Arc;

use super::{
    MetricAttributes, MetricMetadata, NumericAttributes, NumericEntry,
    state::{FormatOptions, NumericMetricState},
};
use crate::metric::{Metric, MetricEntry, MetricName, Numeric};

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

    fn attributes(&self) -> MetricAttributes {
        NumericAttributes {
            unit: None,
            higher_is_better: false,
        }
        .into()
    }
}

impl Numeric for LearningRateMetric {
    fn value(&self) -> NumericEntry {
        self.state.value()
    }
}
