use super::state::FormatOptions;
use super::state::NumericMetricState;
use super::MetricEntry;
use super::MetricMetadata;
use crate::metric::{Metric, Numeric};

/// The loss metric.
#[derive(Default)]
pub struct IterationSpeedMetric {
    state: NumericMetricState,
    instant: Option<std::time::Instant>,
}

impl IterationSpeedMetric {
    /// Create the metric.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Metric for IterationSpeedMetric {
    type Input = ();

    fn update(&mut self, _: &Self::Input, metadata: &MetricMetadata) -> MetricEntry {
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

    fn name(&self) -> String {
        "Iteration Speed".to_string()
    }
}

impl Numeric for IterationSpeedMetric {
    fn value(&self) -> f64 {
        self.state.value()
    }
}
