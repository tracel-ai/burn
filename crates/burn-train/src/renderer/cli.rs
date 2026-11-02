use burn_core::data::dataloader::Progress;

use crate::{
    logger::{EvaluationProgressLogger, ProgressSnapshot, TrainingProgressLogger},
    renderer::{MetricState, MetricsRenderer, MetricsRendererEvaluation, MetricsRendererTraining},
};

/// A simple renderer for when the cli feature is not enabled.
pub struct CliMetricsRenderer {
    label: String,
    training_progress: ProgressSnapshot,
    has_epochs: bool,
    eval_progress: ProgressSnapshot,
}

#[allow(clippy::new_without_default)]
impl CliMetricsRenderer {
    /// Create a new instance.
    pub fn new() -> Self {
        let init = Progress::new(0, 0, Some(String::new()));
        Self {
            training_progress: ProgressSnapshot::new(init.clone(), init.clone()),
            has_epochs: true,
            eval_progress: ProgressSnapshot::new(init.clone(), init),
            label: String::from("training"),
        }
    }
}

impl MetricsRendererTraining for CliMetricsRenderer {
    fn update_train(&mut self, _state: MetricState) {}

    fn update_valid(&mut self, _state: MetricState) {}
}

impl TrainingProgressLogger for CliMetricsRenderer {
    fn start(
        &mut self,
        total_epochs: usize,
        starting_epoch: usize,
        total_items: Option<usize>,
        label: Option<&str>,
    ) {
        self.has_epochs = total_epochs > 0;
        self.training_progress.global =
            Progress::new(starting_epoch, total_epochs, Some("epochs".to_string()));
        if let Some(items) = total_items {
            self.training_progress.split = Progress::new(0, items, Some("items".to_string()));
        }
        self.label = label.unwrap_or("training").to_string();
        println!("Starting {} for {} epochs.", self.label, total_epochs);
    }

    fn start_split(&mut self, split_name: &str, progress: &Progress) {
        self.training_progress.split = progress.clone();
        println!(
            "Starting split '{split_name}': {progress} {}.",
            progress.unit.as_deref().unwrap_or("items")
        );
    }

    fn update_split(&mut self, progress: &Progress) {
        self.training_progress.split = progress.clone();

        // Step-based runs mirror split progress.
        if !self.has_epochs {
            self.training_progress.global = self.training_progress.split.clone();
        }
        println!(
            "{}: {}",
            progress.unit.as_deref().unwrap_or("items"),
            progress
        );
    }

    fn update_epoch(&mut self, epoch: usize) {
        let total = self.training_progress.global.total;
        let unit = self.training_progress.global.unit.clone();
        self.training_progress.global = Progress::new(epoch + 1, total, unit);
    }

    fn end_split(&mut self, progress: &Progress) {
        self.training_progress.split = progress.clone();
        println!("Split ended: {progress}");
    }

    fn end(&mut self) {
        println!("{} ended.", self.label);
    }

    fn log_event_training(&mut self, _event: String) {}
}

impl EvaluationProgressLogger for CliMetricsRenderer {
    fn start_global_progress(&mut self, total_tests: usize) {
        self.eval_progress.global = Progress::new(0, total_tests, Some("tests".to_string()));
        println!("Starting evaluation with {total_tests} test(s).");
    }

    fn start_test(&mut self, name: &str, progress: &Progress) {
        let current = self.eval_progress.global.items_processed + 1;
        let total = self.eval_progress.global.total;
        self.eval_progress.global = Progress::new(current, total, Some("tests".to_string()));
        self.eval_progress.split = progress.clone();
        println!(
            "Starting test '{name}': {progress} {}.",
            progress.unit.as_deref().unwrap_or("items")
        );
    }

    fn update_test_progress(&mut self, progress: &Progress) {
        self.eval_progress.split = progress.clone();
        println!(
            "{}: {}",
            progress.unit.as_deref().unwrap_or("items"),
            progress
        );
    }

    fn end_test(&mut self, progress: &Progress) {
        self.eval_progress.split = progress.clone();
        println!("Split ended: {progress}");
    }

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
