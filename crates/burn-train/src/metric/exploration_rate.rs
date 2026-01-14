use std::sync::Arc;

use super::{
    MetricAttributes, MetricMetadata, NumericAttributes, NumericEntry,
    state::{FormatOptions, NumericMetricState},
};
use crate::metric::{Metric, MetricName, Numeric, SerializedEntry};

/// Metric for the length of the last completed episode.
#[derive(Clone)]
pub struct ExplorationRateMetric {
    name: MetricName,
    state: NumericMetricState,
}

impl ExplorationRateMetric {
    /// Creates a new episode length metric.
    pub fn new() -> Self {
        Self {
            name: Arc::new("Exploration rate".to_string()),
            state: NumericMetricState::new(),
        }
    }
}

impl Default for ExplorationRateMetric {
    fn default() -> Self {
        Self::new()
    }
}

/// The [ExplorationRateMetric](ExplorationRateMetric) input type.
#[derive(new)]
pub struct ExplorationRateInput {
    exploration_rate: f64,
}

impl Metric for ExplorationRateMetric {
    type Input = ExplorationRateInput;

    fn update(
        &mut self,
        item: &ExplorationRateInput,
        _metadata: &MetricMetadata,
    ) -> SerializedEntry {
        self.state.update(
            item.exploration_rate,
            1,
            FormatOptions::new(self.name()).precision(3),
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
            unit: Some(String::from("%")),
            higher_is_better: false,
        }
        .into()
    }
}

impl Numeric for ExplorationRateMetric {
    fn value(&self) -> NumericEntry {
        self.state.current_value()
    }

    fn running_value(&self) -> NumericEntry {
        self.state.running_value()
    }
}
