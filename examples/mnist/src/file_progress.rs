use std::{
    collections::HashMap,
    fs::{File, OpenOptions},
    io::Write,
    path::Path,
};

use burn::train::logger::{EvaluationProgressLogger, TrainingProgressLogger};

/// A progress logger that appends training progress to a file.
///
/// Each event is written as a new line so the complete history across all
/// epochs and splits is preserved. Safe to use alongside a TUI renderer.
///
/// # Example
///
/// ```ignore
/// let training = SupervisedTraining::new(/* ... */)
///     .with_progress_logger(
///         FileTrainingProgressLogger::new("/tmp/my-run/training_progress.log").unwrap()
///     );
/// ```
pub struct FileTrainingProgressLogger {
    writer: File,
    event_counters: HashMap<String, usize>,
}

impl FileTrainingProgressLogger {
    /// Opens (or creates) the file at `path` in append mode.
    pub fn new(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self {
            writer: file,
            event_counters: HashMap::new(),
        })
    }

    fn write(&mut self, line: &str) {
        if let Err(e) = writeln!(self.writer, "{line}") {
            log::warn!("FileTrainingProgressLogger write error: {e}");
        }
    }
}

impl TrainingProgressLogger for FileTrainingProgressLogger {
    fn start(&mut self, total_epochs: usize, total_items: Option<usize>) {
        match total_items {
            Some(n) => self.write(&format!(
                "[Training] start  epochs={total_epochs} total_items={n}"
            )),
            None => self.write(&format!("[Training] start  epochs={total_epochs}")),
        }
    }

    fn update_epoch(&mut self, epoch: usize) {
        self.write(&format!("[Training] epoch_complete  epoch={epoch}"));
    }

    fn start_split(&mut self, split: &str, total_items: usize) {
        self.write(&format!(
            "[Training] split_start  split={split} total_items={total_items}"
        ));
    }

    fn update_split(&mut self, items_processed: usize) {
        self.write(&format!(
            "[Training] split_update  items_processed={items_processed}"
        ));
    }

    fn end_split(&mut self) {
        self.write("[Training] split_end");
        self.event_counters.values_mut().for_each(|v| *v = 0);
    }

    fn end(&mut self) {
        self.write("[Training] end");
    }

    fn log_event_training(&mut self, event: String) {
        let count = {
            let c = self.event_counters.entry(event.clone()).or_insert(0);
            *c += 1;
            *c
        };
        self.write(&format!("[event] {event} = {count}"));
    }
}

/// A progress logger that appends evaluation progress to a file.
///
/// Each event is written as a new line so the complete history across all
/// test splits is preserved. Safe to use alongside a TUI renderer.
///
/// # Example
///
/// ```ignore
/// let evaluator = EvaluatorBuilder::new(/* ... */)
///     .with_progress_logger(
///         FileEvaluationProgressLogger::new("/tmp/my-run/evaluation_progress.log").unwrap()
///     );
/// ```
pub struct FileEvaluationProgressLogger {
    writer: File,
    event_counters: HashMap<String, usize>,
}

impl FileEvaluationProgressLogger {
    /// Opens (or creates) the file at `path` in append mode.
    pub fn new(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self {
            writer: file,
            event_counters: HashMap::new(),
        })
    }

    fn write(&mut self, line: &str) {
        if let Err(e) = writeln!(self.writer, "{line}") {
            log::warn!("FileEvaluationProgressLogger write error: {e}");
        }
    }
}

impl EvaluationProgressLogger for FileEvaluationProgressLogger {
    fn start_global_progress(&mut self, total_tests: usize) {
        self.write(&format!("[Evaluation] start  total_tests={total_tests}"));
    }

    fn start_test(&mut self, name: &str, total_items: usize) {
        self.write(&format!(
            "[Evaluation] test_start  name={name} total_items={total_items}"
        ));
    }

    fn update_test_progress(&mut self, items_processed: usize) {
        self.write(&format!(
            "[Evaluation] test_update  items_processed={items_processed}"
        ));
    }

    fn end_test(&mut self) {
        self.write("[Evaluation] test_end");
        self.event_counters.values_mut().for_each(|v| *v = 0);
    }

    fn end_global_progress(&mut self) {
        self.write("[Evaluation] end");
    }

    fn log_event_evaluation(&mut self, event: String) {
        let count = {
            let c = self.event_counters.entry(event.clone()).or_insert(0);
            *c += 1;
            *c
        };
        self.write(&format!("[event] {event} = {count}"));
    }
}
