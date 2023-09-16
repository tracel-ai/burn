use crate::metric::dashboard::{DashboardMetricState, DashboardRenderer, TrainingProgress};

/// A simple renderer for when the cli feature is not enabled.
pub struct CLIDashboardRenderer;

impl CLIDashboardRenderer {
    /// Create a new instance.
    pub fn new() -> Self {
        Self {}
    }
}

impl DashboardRenderer for CLIDashboardRenderer {
    fn update_train(&mut self, _state: DashboardMetricState) {}

    fn update_valid(&mut self, _state: DashboardMetricState) {}

    fn render_train(&mut self, item: TrainingProgress) {
        dbg!(item);
    }

    fn render_valid(&mut self, item: TrainingProgress) {
        dbg!(item);
    }
}
