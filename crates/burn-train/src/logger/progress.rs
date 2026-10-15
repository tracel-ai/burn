use crate::SplitOutcome;
use burn_core::data::dataloader::Progress;

/// Trait for logging training progress at each step and end of epoch.
///
/// # Call sequence
///
/// Implementors can expect the following sequence of calls for a complete training run:
///
/// ```text
/// start(total_epochs, starting_epoch, total_items, label)
///   for each epoch:
///     start_split("train", progress)
///       update_split(progress)  // called once per batch
///       ...
///     end_split(progress, outcome)
///     start_split("valid", progress)
///       update_split(progress)  // called once per batch
///       ...
///     end_split(progress, outcome)
///     update_epoch(epoch)
/// end()
/// ```
///
/// `end()` is called whether training completes normally or is interrupted early.
///
/// Split updates carry the current count, expected total, and unit. Terminal callbacks carry the outcome.
pub trait TrainingProgressLogger: Send {
    /// Called once at the start of training, providing the total number of epochs and the starting epoch.
    ///
    /// The total number of items of the training can optionally be provided if it is known.
    fn start(
        &mut self,
        total_epochs: usize,
        starting_epoch: usize,
        total_items: Option<usize>,
        label: Option<&str>,
    );

    /// Called at the end of each full epoch (after both train and valid splits complete).
    fn update_epoch(&mut self, epoch: usize);

    /// Called at the start of a training split, providing the split name and initial progress.
    fn start_split(&mut self, split: &str, progress: &Progress);

    /// Log a progress snapshot for the current split.
    fn update_split(&mut self, progress: &Progress);

    /// Called once at the end of a training split with final measurements and its outcome.
    ///
    /// This also applies to empty, interrupted, and failed splits. A final snapshot does not
    /// produce an iteration event or an additional `update_split` call.
    fn end_split(&mut self, progress: &Progress, outcome: SplitOutcome);

    /// Called at the end of training, whether it completed successfully or was interrupted.
    fn end(&mut self);

    /// Log a custom named event that falls outside the standard training lifecycle callbacks.
    fn log_event_training(&mut self, event: String);
}

/// Trait for logging evaluation progress at each step and end of evaluation.
///
/// # Call sequence
///
/// Implementors can expect the following sequence of calls for a complete evaluation run:
///
/// ```text
/// start_global_progress(total_tests)
///   for each test split:
///     start_test(name, progress)
///       update_test_progress(progress)  // called once per batch
///       ...
///     end_test(progress, outcome)
/// end_global_progress()
/// ```
///
/// `end_global_progress()` is called whether evaluation completes normally or is interrupted early.
///
/// Test updates carry the current count, expected total, and unit. Terminal callbacks carry the outcome.
pub trait EvaluationProgressLogger: Send {
    /// Called once at the start of evaluation, providing the total number of test splits.
    fn start_global_progress(&mut self, total_tests: usize);

    /// Called at the start of a test split, providing the split name and initial progress.
    fn start_test(&mut self, name: &str, progress: &Progress);

    /// Log a progress snapshot for the current test split.
    fn update_test_progress(&mut self, progress: &Progress);

    /// Called once at the end of a test split with final measurements and its outcome.
    ///
    /// This also applies to empty, interrupted, and failed splits. A final snapshot does not
    /// produce an iteration event or an additional `update_test_progress` call.
    fn end_test(&mut self, progress: &Progress, outcome: SplitOutcome);

    /// Called at the end of evaluation.
    fn end_global_progress(&mut self);

    /// Log a custom named event that falls outside the standard evaluation lifecycle callbacks.
    fn log_event_evaluation(&mut self, event: String);
}

/// Two-level progress snapshot combining run-level and phase-level tracking.
///
/// `global_progress` spans the full training run (e.g., epochs completed out of total),
/// while `split_progress` tracks the current phase (e.g., batches within the current epoch).
#[derive(Debug, Clone)]
pub struct ProgressSnapshot {
    /// Progress across the entire training run.
    pub global: Progress,
    /// Progress within the current phase (epoch or evaluation split).
    pub split: Progress,
}

impl ProgressSnapshot {
    /// Create a new overall progress snapshot.
    pub fn new(global: Progress, split: Progress) -> Self {
        Self { global, split }
    }
}
