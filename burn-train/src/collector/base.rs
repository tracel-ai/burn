use burn_core::{data::dataloader::Progress, LearningRate};

/// Event happening during the training process.
pub enum TrainingEvent<T> {
    /// Processing an item for either training of inference.
    ProcessedItem(LearnerItem<T>),
    /// Signal the end of an epoch.
    EndEpoch(usize),
}

/// The base trait for trainer callbacks.
pub trait TrainingEventCollector: Send {
    /// Training item.
    type ItemTrain;
    /// Validation item.
    type ItemValid;

    fn on_event_train(&mut self, event: TrainingEvent<Self::ItemTrain>);
    fn on_event_valid(&mut self, event: TrainingEvent<Self::ItemValid>);

    /// Find the epoch following the given criteria following the collected data.
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
