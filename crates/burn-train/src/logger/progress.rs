use std::{fs::File, io::Write, path::Path};

use crate::renderer::{EvaluationProgress, TrainingProgress};

/// Trait for logging training progress at each step and end of epoch.
pub trait TrainingProgressLogger: Send {
    /// Log the progress of the current training step.
    fn update_train(&mut self, progress: &TrainingProgress);

    /// Log the progress of the current validation step.
    fn update_valid(&mut self, progress: &TrainingProgress);

    /// Called at the end of an epoch evaluation.
    fn end_epoch(&mut self, epoch: usize);
}

/// Trait for logging evaluation progress at each step and end of evaluation.
pub trait EvaluationProgressLogger: Send {
    /// Log the progress of the current test step.
    fn update_test(&mut self, progress: &EvaluationProgress);

    /// Called at the end of the evaluation.
    fn end_eval(&mut self);
}

/// A simple file-based implementation of [TrainingProgressLogger] and [EvaluationProgressLogger] for debugging.
pub struct FileProgressLogger {
    file: File,
}

impl FileProgressLogger {
    /// Create a new file progress logger writing to the given path.
    pub fn new(path: impl AsRef<Path>) -> Self {
        let file = File::create(path).expect("Should be able to create progress log file.");
        Self { file }
    }
}

impl TrainingProgressLogger for FileProgressLogger {
    fn update_train(&mut self, progress: &TrainingProgress) {
        let items = progress
            .progress
            .as_ref()
            .map(|p| format!("{}/{}", p.items_processed, p.items_total))
            .unwrap_or_else(|| "?".to_string());

        writeln!(
            self.file,
            "[TRAIN] epoch: {}/{} | items: {} | iter: {}",
            progress.global_progress.items_processed,
            progress.global_progress.items_total,
            items,
            progress
                .iteration
                .map_or("?".to_string(), |i| i.to_string()),
        )
        .ok();
    }

    fn update_valid(&mut self, progress: &TrainingProgress) {
        let items = progress
            .progress
            .as_ref()
            .map(|p| format!("{}/{}", p.items_processed, p.items_total))
            .unwrap_or_else(|| "?".to_string());

        writeln!(
            self.file,
            "[VALID] epoch: {}/{} | items: {} | iter: {}",
            progress.global_progress.items_processed,
            progress.global_progress.items_total,
            items,
            progress
                .iteration
                .map_or("?".to_string(), |i| i.to_string()),
        )
        .ok();
    }

    fn end_epoch(&mut self, epoch: usize) {
        writeln!(self.file, "[END_EPOCH] epoch: {}", epoch).ok();
    }
}

impl EvaluationProgressLogger for FileProgressLogger {
    fn update_test(&mut self, progress: &EvaluationProgress) {
        writeln!(
            self.file,
            "[TEST] items: {}/{} | iter: {}",
            progress.progress.items_processed,
            progress.progress.items_total,
            progress
                .iteration
                .map_or("?".to_string(), |i| i.to_string()),
        )
        .ok();
    }

    fn end_eval(&mut self) {
        writeln!(self.file, "[END_EVAL]").ok();
    }
}
