use crate::{
    logger::{EvaluationProgressLogger, TrainingProgressLogger},
    renderer::{
        EvaluationProgress, MetricState, MetricsRenderer, MetricsRendererEvaluation,
        MetricsRendererTraining, ProgressType, TrainingProgress,
    },
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

    fn update_split(&mut self, item: &TrainingProgress, _progress_indicators: Vec<ProgressType>) {
        println!("{item:?}");
    }

    fn update_epoch(&mut self, epoch: usize) {
        todo!()
    }

    fn end_split(&mut self) {
        todo!()
    }

    fn end(&mut self) {
        todo!()
    }
}

impl EvaluationProgressLogger for CliMetricsRenderer {
    fn start(&mut self, total_tests: usize) {
        todo!()
    }

    fn start_test(&mut self, name: &str, total_items: usize) {
        todo!()
    }

    fn update_test_progress(
        &mut self,
        progress: &EvaluationProgress,
        indicators: Vec<ProgressType>,
    ) {
        todo!()
    }

    fn end_test(&mut self) {
        todo!()
    }

    fn end(&mut self) {
        todo!()
    }
}

impl MetricsRendererEvaluation for CliMetricsRenderer {
    fn update_test(&mut self, _name: super::EvaluationName, _state: MetricState) {}
}

impl MetricsRenderer for CliMetricsRenderer {
    fn manual_close(&mut self) {
        // Nothing to do.
    }

    fn register_metric(&mut self, _definition: crate::metric::MetricDefinition) {}
}
