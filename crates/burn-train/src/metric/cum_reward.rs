use std::sync::Arc;

use super::{
    MetricAttributes, MetricMetadata, NumericAttributes, NumericEntry,
    state::{FormatOptions, NumericMetricState},
};
use crate::metric::{Metric, MetricName, Numeric, SerializedEntry};

/// Metric for the cumulative reward of the last completed episode.
#[derive(Clone)]
pub struct CumulativeRewardMetric {
    name: MetricName,
    state: NumericMetricState,
}

impl CumulativeRewardMetric {
    /// Creates a new episode length metric.
    pub fn new() -> Self {
        Self {
            name: Arc::new("Cum. Reward".to_string()),
            state: NumericMetricState::new(),
        }
    }
}

impl Default for CumulativeRewardMetric {
    fn default() -> Self {
        Self::new()
    }
}

/// The [CumulativeRewardMetric](CumulativeRewardMetric) input type.
#[derive(new)]
pub struct CumulativeRewardInput {
    cum_reward: f64,
}

impl Metric for CumulativeRewardMetric {
    type Input = CumulativeRewardInput;

    fn update(
        &mut self,
        item: &CumulativeRewardInput,
        _metadata: &MetricMetadata,
    ) -> SerializedEntry {
        self.state.update(
            item.cum_reward,
            1,
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
            higher_is_better: true,
        }
        .into()
    }
}

impl Numeric for CumulativeRewardMetric {
    fn value(&self) -> NumericEntry {
        self.state.current_value()
    }

    fn running_value(&self) -> NumericEntry {
        self.state.running_value()
    }
}
