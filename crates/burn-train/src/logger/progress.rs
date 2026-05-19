/// Trait for logging training progress at each step and end of epoch.
///
/// TODO: document how the trait caller is expected to call these methods in a way that a single epoch and split are started and ended exactly once.
pub trait TrainingProgressLogger: Send {
    /// Called once at the start of training, providing the total number of epochs.
    ///
    /// The total number of items of the training can optionally be provided if it is known.
    fn start(&mut self, total_epochs: usize, total_items: Option<usize>);

    /// Called at the end of each epoch, providing the epoch number.
    fn update_epoch(&mut self, epoch: usize);

    /// Called at the start of a training split, providing the split name and total number of items.
    fn start_split(&mut self, split: String, total_items: usize);

    /// Log the progress of the current training step.
    fn update_split(&mut self, items_processed: usize);

    /// Called at the end of a training split.
    fn end_split(&mut self);

    /// Called at the end of training, whether it completed successfully or was interrupted.
    fn end(&mut self);
}

/// Trait for logging evaluation progress at each step and end of evaluation.
///
/// TODO: document how the trait caller is expected to call these methods in a way that a single evaluation and test split are started and ended exactly once.
pub trait EvaluationProgressLogger: Send {
    /// Called once at the start of evaluation, providing the total number of test splits.
    fn start(&mut self, total_tests: usize);

    /// Called at the start of a test split, providing the split name and total.
    fn start_test(&mut self, name: String, total_items: usize);

    /// Log the progress of the current test step.
    fn update_test(&mut self, items_processed: usize);

    /// Called at the end of a test split.
    fn end_test(&mut self);

    /// Called at the end of evaluation.
    fn end(&mut self);
}
