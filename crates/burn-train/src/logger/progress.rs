use std::{fs::File, io::Write, path::Path};

use crate::renderer::TrainingProgress;

/// Trait for logging training progress at each step and end of epoch.
pub trait TrainingProgressLogger: Send {
    /// Log the progress of the current training step.
    fn update_train(&mut self, progress: &TrainingProgress);

    /// Log the progress of the current validation step.
    fn update_valid(&mut self, progress: &TrainingProgress);

    /// Called at the end of an epoch evaluation.
    fn end_evaluation(&mut self, epoch: usize);
}

/// A simple file-based implementation of [TrainingProgressLogger] for debugging.
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
            progress.iteration.map_or("?".to_string(), |i| i.to_string()),
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
            progress.iteration.map_or("?".to_string(), |i| i.to_string()),
        )
        .ok();
    }

    fn end_evaluation(&mut self, epoch: usize) {
        writeln!(self.file, "[END_EPOCH] epoch: {}", epoch).ok();
    }
}
