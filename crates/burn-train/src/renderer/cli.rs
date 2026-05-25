use crate::{
    logger::{EvaluationProgressLogger, OverallProgress, ProgressEvent, TrainingProgressLogger},
    renderer::{MetricState, MetricsRenderer, MetricsRendererEvaluation, MetricsRendererTraining},
};

/// A simple renderer for when the cli feature is not enabled.
pub struct CliMetricsRenderer;

#[allow(clippy::new_without_default)]
impl CliMetricsRenderer {
    /// Create a new instance.
    pub fn new() -> Self {
        Self {}
    }
}

impl MetricsRendererTraining for CliMetricsRenderer {
    fn update_train(&mut self, _state: MetricState) {}

    fn update_valid(&mut self, _state: MetricState) {}
}

impl TrainingProgressLogger for CliMetricsRenderer {
    fn start(&mut self, total_epochs: usize, _total_items: Option<usize>) {
        println!("Starting training for {total_epochs} epochs.");
    }

    fn start_split(&mut self, split_name: &str, total_items: usize) {
        println!("Starting split '{split_name}' with {total_items} items.");
    }

    fn update_split(&mut self, item: &OverallProgress) {
        println!("{item:?}");
    }

    fn update_epoch(&mut self, _epoch: usize) {}

    fn end_split(&mut self) {
        println!("Split ended.");
    }

    fn end(&mut self) {
        println!("Training ended.");
    }

    fn log_event(&mut self, _event: ProgressEvent) {}
}

impl EvaluationProgressLogger for CliMetricsRenderer {
    fn start_global_progress(&mut self, total_tests: usize) {
        println!("Starting evaluation with {total_tests} test(s).");
    }

    fn start_test(&mut self, name: &str, total_items: usize) {
        println!("Starting test '{name}' with {total_items} items.");
    }

    fn update_test_progress(&mut self, progress: &OverallProgress) {
        println!("{progress:?}");
    }

    fn end_test(&mut self) {}

    fn end_global_progress(&mut self) {}
}

impl MetricsRendererEvaluation for CliMetricsRenderer {
    fn update_test(&mut self, _name: super::EvaluationName, _state: MetricState) {}
}

impl MetricsRenderer for CliMetricsRenderer {
    fn manual_close(&mut self) {}

    fn register_metric(&mut self, _definition: crate::metric::MetricDefinition) {}
}
