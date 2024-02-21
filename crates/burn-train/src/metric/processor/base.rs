use burn_core::data::dataloader::Progress;
use burn_core::LearningRate;

/// Event happening during the training/validation process.
pub enum Event<T> {
    /// Signal that an item have been processed.
    ProcessedItem(LearnerItem<T>),
    /// Signal the end of an epoch.
    EndEpoch(usize),
}

/// Process events happening during training and validation.
pub trait EventProcessor {
    /// The training item.
    type ItemTrain;
    /// The validation item.
    type ItemValid;

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
