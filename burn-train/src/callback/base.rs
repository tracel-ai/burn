use burn_core::{data::dataloader::Progress, LearningRate};

/// The base trait for trainer callbacks.
pub trait LearnerCallback: Send {
    /// Training item.
    type ItemTrain;
    /// Validation item.
    type ItemValid;

    /// Called when a training item is logged.
    fn on_train_item(&mut self, _item: LearnerItem<Self::ItemTrain>) {}

    /// Called when a validation item is logged.
    fn on_valid_item(&mut self, _item: LearnerItem<Self::ItemValid>) {}

    /// Called when a training epoch is finished.
    fn on_train_end_epoch(&mut self, _epoch: usize) {}

    /// Called when a validation epoch is finished.
    fn on_valid_end_epoch(&mut self, _epoch: usize) {}

    /// Find the epoch following the given criteria.
    fn find_epoch(
        &mut self,
        name: &str,
        aggregate: Aggregate,
        direction: Direction,
        split: Split,
    ) -> Option<usize>;
}

/// How to aggregate the metric.
pub enum Aggregate {
    /// Compute the average.
    Mean,
}

/// The split to use.
pub enum Split {
    /// The training split.
    Train,
    /// The validation split.
    Valid,
}
/// The direction of the query.
pub enum Direction {
    /// Lower is better.
    Lowest,
    /// Higher is better.
    Hightest,
}

/// A learner item.
#[derive(new)]
pub struct LearnerItem<T> {
    /// The item.
    pub item: T,

    /// The progress.
    pub progress: Progress,

    /// The epoch.
    pub epoch: usize,

    /// The total number of epochs.
    pub epoch_total: usize,

    /// The iteration.
    pub iteration: usize,

    /// The learning rate.
    pub lr: Option<LearningRate>,
}
