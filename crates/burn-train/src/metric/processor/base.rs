use burn_core::LearningRate;
use burn_core::data::dataloader::Progress;

/// Event happening during the training/validation process.
pub enum Event<T> {
    /// Signal that an item have been processed.
    ProcessedItem(LearnerItem<T>),
    /// Signal the end of an epoch.
    EndEpoch(usize),
    /// Signal the end of the process (e.g., training end).
    End,
}

/// Items that are lazy are not ready to be processed by metrics.
///
/// We want to sync them on a different thread to avoid blocking training.
pub trait ItemLazy: Send {
    /// Item that is properly synced and ready to be processed by metrics.
    type ItemSync: Send;

    /// Sync the item.
    fn sync(self) -> Self::ItemSync;
}

/// Process events happening during training and validation.
pub trait EventProcessor: Send {
    /// The training item.
    type ItemTrain: ItemLazy;
    /// The validation item.
    type ItemValid: ItemLazy;

    /// Collect a training event.
    fn process_train(&mut self, event: Event<Self::ItemTrain>);
    /// Collect a validation event.
    fn process_valid(&mut self, event: Event<Self::ItemValid>);
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

impl<T: ItemLazy> ItemLazy for LearnerItem<T> {
    type ItemSync = LearnerItem<T::ItemSync>;

    fn sync(self) -> Self::ItemSync {
        LearnerItem {
            item: self.item.sync(),
            progress: self.progress,
            epoch: self.epoch,
            epoch_total: self.epoch_total,
            iteration: self.iteration,
            lr: self.lr,
        }
    }
}
