use std::sync::Arc;

use super::MetricMetadata;
use super::SerializedEntry;
use super::state::FormatOptions;
use super::state::NumericMetricState;
use crate::metric::MetricName;
use crate::metric::Numeric;
use crate::metric::{Metric, MetricAttributes, NumericAttributes, NumericEntry};

/// The loss metric.
#[derive(Clone)]
pub struct IterationSpeedMetric {
    name: MetricName,
    state: NumericMetricState,
    instant: Option<std::time::Instant>,
}

impl Default for IterationSpeedMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl IterationSpeedMetric {
    /// Create the metric.
    pub fn new() -> Self {
        Self {
            name: Arc::new("Iteration Speed".to_string()),
            state: Default::default(),
            instant: Default::default(),
        }
    }
}

impl Metric for IterationSpeedMetric {
    type Input = ();

    fn update(&mut self, _: &Self::Input, metadata: &MetricMetadata) -> SerializedEntry {
        let raw = match self.instant {
            Some(val) => metadata.iteration as f64 / val.elapsed().as_secs_f64(),
            None => {
                self.instant = Some(std::time::Instant::now());
                0.0
            }
        };

        self.state.update(
            raw,
            1,
            FormatOptions::new(self.name())
                .unit("iter/sec")
                .precision(2),
        )
    }

    fn clear(&mut self) {
        self.instant = None;
    }

    fn name(&self) -> MetricName {
        self.name.clone()
    }

    fn attributes(&self) -> MetricAttributes {
        NumericAttributes {
            unit: Some("iter/sec".to_string()),
            higher_is_better: true,
        }
        .into()
    }
}

impl Numeric for IterationSpeedMetric {
    fn value(&self) -> NumericEntry {
        self.state.current_value()
    }

    fn running_value(&self) -> NumericEntry {
        self.state.running_value()
    }
}
