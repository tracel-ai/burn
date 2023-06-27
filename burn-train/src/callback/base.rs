use burn_core::{data::dataloader::Progress, LearningRate};

/// The base trait for trainer callbacks.
pub trait LearnerCallback<T, V>: Send {
    /// Called when a training item is logged.
    fn on_train_item(&mut self, _item: LearnerItem<T>) {}

    /// Called when a validation item is logged.
    fn on_valid_item(&mut self, _item: LearnerItem<V>) {}

    /// Called when a training epoch is finished.
    fn on_train_end_epoch(&mut self, _epoch: usize) {}

    /// Called when a validation epoch is finished.
    fn on_valid_end_epoch(&mut self, _epoch: usize) {}
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
