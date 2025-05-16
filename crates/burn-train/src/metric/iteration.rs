use super::MetricEntry;
use super::MetricMetadata;
use super::state::FormatOptions;
use super::state::NumericMetricState;
use crate::metric::{Metric, Numeric};

/// The loss metric.
#[derive(Default)]
pub struct IterationSpeedMetric {
    state: NumericMetricState,
    instant: Option<std::time::Instant>,
}

impl IterationSpeedMetric {
    /// Create the metric.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl Metric for IterationSpeedMetric {
    type Input = ();

    fn update(&mut self, (): &Self::Input, metadata: &MetricMetadata) -> MetricEntry {
        let raw = if let Some(val) = self.instant {
            metadata.iteration as f64 / val.elapsed().as_secs_f64()
        } else {
            self.instant = Some(std::time::Instant::now());
            0.0
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
