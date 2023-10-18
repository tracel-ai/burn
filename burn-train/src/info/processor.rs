use crate::LearnerItem;

/// Event happening during the training/validation process.
pub enum Event<T> {
    /// Signal that an item have been processed.
    ProcessedItem(LearnerItem<T>),
    /// Signal the end of an epoch.
    EndEpoch(usize),
}

pub trait EventProcessor {
    type ItemTrain;
    type ItemValid;

    /// Collect the training event.
    fn add_event_train(&mut self, event: Event<Self::ItemTrain>);
    fn add_event_valid(&mut self, event: Event<Self::ItemValid>);
}
