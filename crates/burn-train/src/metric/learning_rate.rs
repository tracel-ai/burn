use std::sync::Arc;

use super::{
    MetricAttributes, MetricMetadata, NumericAttributes, NumericEntry,
    state::{FormatOptions, NumericMetricState},
};
use crate::metric::{Metric, MetricName, Numeric, SerializedEntry};

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

    fn update(&mut self, _item: &(), metadata: &MetricMetadata) -> SerializedEntry {
        // TODO: We only log the default learning rate. Yet another motivation to introduce metric groups.
        let lr = metadata.lr.as_ref().map(|val| val.base()).unwrap_or(0.0);

        self.state.update(lr, 1);
        self.state
            .compute_update(FormatOptions::new(self.name()).precision(2))
    }

    fn compute(&mut self) -> SerializedEntry {
        self.state
            .compute_final(FormatOptions::new(self.name()).precision(2))
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

// TODO: LR should probably just report the current value, the aggregated values don't make as much sense esp. for visualization
impl Numeric for LearningRateMetric {
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
