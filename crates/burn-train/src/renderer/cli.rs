use crate::renderer::{MetricState, MetricsRenderer, TrainingProgress};

/// A simple renderer for when the cli feature is not enabled.
pub struct CliMetricsRenderer;

#[allow(clippy::new_without_default)]
impl CliMetricsRenderer {
    /// Create a new instance.
    pub fn new() -> Self {
        Self {}
    }
}

impl MetricsRenderer for CliMetricsRenderer {
    fn update_train(&mut self, _state: MetricState) {}

    fn update_valid(&mut self, _state: MetricState) {}

    fn render_train(&mut self, item: TrainingProgress) {
        println!("{:?}", item);
    }

    fn render_valid(&mut self, item: TrainingProgress) {
        println!("{:?}", item);
    }
}
