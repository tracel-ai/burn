use crate::renderer::OverallProgress;

/// Trait for logging training progress at each step and end of epoch.
///
/// # Call sequence
///
/// Implementors can expect the following sequence of calls for a complete training run:
///
/// ```text
/// start(total_epochs, total_items)
///   for each epoch:
///     start_split("train", total_items_train)
///       update_split(progress)  // called once per batch
///       ...
///     end_split()
///     start_split("valid", total_items_valid)
///       update_split(progress)  // called once per batch
///       ...
///     end_split()
///     update_epoch(epoch)
/// end()
/// ```
///
/// `end()` is called whether training completes normally or is interrupted early.
pub trait TrainingProgressLogger: Send {
    /// Called once at the start of training, providing the total number of epochs.
    ///
    /// The total number of items of the training can optionally be provided if it is known.
    fn start(&mut self, total_epochs: usize, total_items: Option<usize>);

    /// Called at the end of each full epoch (after both train and valid splits complete).
    fn update_epoch(&mut self, epoch: usize);

    /// Called at the start of a training split, providing the split name and total number of items.
    fn start_split(&mut self, split: &str, total_items: usize);

    /// Log the progress of the current training step.
    fn update_split(&mut self, progress: &OverallProgress);

    /// Called at the end of a training split.
    fn end_split(&mut self);

    /// Called at the end of training, whether it completed successfully or was interrupted.
    fn end(&mut self);
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
///     start_test(name, total_items)
///       update_test_progress(progress)  // called once per batch
///       ...
///     end_test()
/// end_global_progress()
/// ```
///
/// `end()` is called whether evaluation completes normally or is interrupted early.
pub trait EvaluationProgressLogger: Send {
    /// Called once at the start of evaluation, providing the total number of test splits.
    fn start_global_progress(&mut self, total_tests: usize);

    /// Called at the start of a test split, providing the split name and total number of items.
    fn start_test(&mut self, name: &str, total_items: usize);

    /// Log the progress of the current test step.
    fn update_test_progress(&mut self, progress: &OverallProgress);

    /// Called at the end of a test split.
    fn end_test(&mut self);

    /// Called at the end of evaluation.
    fn end_global_progress(&mut self);
}
