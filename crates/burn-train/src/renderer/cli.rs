use burn_core::data::dataloader::Progress;

use crate::{
    logger::{EvaluationProgressLogger, ProgressSnapshot, TrainingProgressLogger},
    renderer::{MetricState, MetricsRenderer, MetricsRendererEvaluation, MetricsRendererTraining},
};

/// A simple renderer for when the cli feature is not enabled.
pub struct CliMetricsRenderer {
    training_progress: ProgressSnapshot,
    eval_progress: ProgressSnapshot,
}

#[allow(clippy::new_without_default)]
impl CliMetricsRenderer {
    /// Create a new instance.
    pub fn new() -> Self {
        let init = Progress::new(0, 0, Some(String::new()));
        Self {
            training_progress: ProgressSnapshot::new(init.clone(), init.clone()),
            eval_progress: ProgressSnapshot::new(init.clone(), init),
        }
    }
}

impl MetricsRendererTraining for CliMetricsRenderer {
    fn update_train(&mut self, _state: MetricState) {}

    fn update_valid(&mut self, _state: MetricState) {}
}

impl TrainingProgressLogger for CliMetricsRenderer {
    fn start(&mut self, total_epochs: usize, total_items: Option<usize>) {
        self.training_progress.global = Progress::new(1, total_epochs, Some("epochs".to_string()));
        if let Some(items) = total_items {
            self.training_progress.split = Progress::new(0, items, Some("items".to_string()));
        }
        println!("Starting training for {total_epochs} epochs.");
    }

    fn start_split(&mut self, split_name: &str, total_items: usize) {
        self.training_progress.split = Progress::new(0, total_items, Some("items".to_string()));
        println!("Starting split '{split_name}' with {total_items} items.");
    }

    fn update_split(&mut self, items_processed: usize) {
        let total = self.training_progress.split.items_total;
        let unit = self.training_progress.split.unit.clone();
        self.training_progress.split = Progress::new(items_processed, total, unit);

        // For RL: global_progress.items_total == 0 means no epoch concept; mirror split.
        if self.training_progress.global.items_total == 0 {
            self.training_progress.global = self.training_progress.split.clone();
        }
        println!("{:?}", self.training_progress);
    }

    fn update_epoch(&mut self, epoch: usize) {
        let total = self.training_progress.global.items_total;
        let unit = self.training_progress.global.unit.clone();
        self.training_progress.global = Progress::new(epoch + 1, total, unit);
    }

    fn end_split(&mut self) {
        println!("Split ended.");
    }

    fn end(&mut self) {
        println!("Training ended.");
    }

    fn log_event_training(&mut self, _event: String) {}
}

impl EvaluationProgressLogger for CliMetricsRenderer {
    fn start_global_progress(&mut self, total_tests: usize) {
        self.eval_progress.global = Progress::new(0, total_tests, Some("tests".to_string()));
        println!("Starting evaluation with {total_tests} test(s).");
    }

    fn start_test(&mut self, name: &str, total_items: usize) {
        let current = self.eval_progress.global.items_processed + 1;
        let total = self.eval_progress.global.items_total;
        self.eval_progress.global = Progress::new(current, total, Some("tests".to_string()));
        self.eval_progress.split = Progress::new(0, total_items, Some("items".to_string()));
        println!("Starting test '{name}' with {total_items} items.");
    }

    fn update_test_progress(&mut self, items_processed: usize) {
        let total = self.eval_progress.split.items_total;
        let unit = self.eval_progress.split.unit.clone();
        self.eval_progress.split = Progress::new(items_processed, total, unit);
        println!("{:?}", self.eval_progress);
    }

    fn end_test(&mut self) {}

    fn end_global_progress(&mut self) {}

    fn log_event_evaluation(&mut self, _event: String) {}
}

impl MetricsRendererEvaluation for CliMetricsRenderer {
    fn update_test(&mut self, _name: super::EvaluationName, _state: MetricState) {}
}

impl MetricsRenderer for CliMetricsRenderer {
    fn manual_close(&mut self) {}

    fn register_metric(&mut self, _definition: crate::metric::MetricDefinition) {}
}
