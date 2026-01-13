use std::sync::Arc;

use super::{
    MetricAttributes, MetricMetadata, NumericAttributes, NumericEntry,
    state::{FormatOptions, NumericMetricState},
};
use crate::metric::{Metric, MetricName, Numeric, SerializedEntry};

/// Metric for the length of the last completed episode.
#[derive(Clone)]
pub struct EpisodeLengthMetric {
    name: MetricName,
    state: NumericMetricState,
}

impl EpisodeLengthMetric {
    /// Creates a new episode length metric.
    pub fn new() -> Self {
        Self {
            name: Arc::new("Episode length".to_string()),
            state: NumericMetricState::new(),
        }
    }
}

impl Default for EpisodeLengthMetric {
    fn default() -> Self {
        Self::new()
    }
}

pub struct EpisodeLengthInput {
    pub ep_len: f64,
}

impl Metric for EpisodeLengthMetric {
    type Input = EpisodeLengthInput;

    fn update(&mut self, item: &EpisodeLengthInput, _metadata: &MetricMetadata) -> SerializedEntry {
        self.state
            .update(item.ep_len, 1, FormatOptions::new(self.name()).precision(0))
    }

    fn clear(&mut self) {
        self.state.reset()
    }

    fn name(&self) -> MetricName {
        self.name.clone()
    }

    fn attributes(&self) -> MetricAttributes {
        NumericAttributes {
            unit: Some(String::from("steps")),
            higher_is_better: true,
        }
        .into()
    }
}

impl Numeric for EpisodeLengthMetric {
    fn value(&self) -> NumericEntry {
        self.state.current_value()
    }

    fn running_value(&self) -> NumericEntry {
        self.state.running_value()
    }
}
