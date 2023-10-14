use burn_core::{data::dataloader::Progress, LearningRate};

/// Event happening during the training/validation process.
pub enum Event<T> {
    /// Signal that an item have been processed.
    ProcessedItem(LearnerItem<T>),
    /// Signal the end of an epoch.
    EndEpoch(usize),
}

/// Defines how training and validation events are collected.
///
/// This trait also exposes methods that uses the collected data to compute useful information.
pub trait EventCollector: Send {
    /// Training item.
    type ItemTrain;
    /// Validation item.
    type ItemValid;

    /// Collect the training event.
    fn on_event_train(&mut self, event: Event<Self::ItemTrain>);

    /// Collect the validaion event.
    fn on_event_valid(&mut self, event: Event<Self::ItemValid>);

    /// Find the epoch following the given criteria from the collected data.
    fn find_epoch(
        &mut self,
        name: &str,
        aggregate: Aggregate,
        direction: Direction,
        split: Split,
    ) -> Option<usize>;
}

#[derive(Copy, Clone)]
/// How to aggregate the metric.
pub enum Aggregate {
    /// Compute the average.
    Mean,
}

#[derive(Copy, Clone)]
/// The split to use.
pub enum Split {
    /// The training split.
    Train,
    /// The validation split.
    Valid,
}

#[derive(Copy, Clone)]
/// The direction of the query.
pub enum Direction {
    /// Lower is better.
    Lowest,
    /// Higher is better.
    Highest,
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

#[cfg(test)]
pub mod test_utils {
    use crate::{info::MetricsInfo, Aggregate, Direction, Event, EventCollector, Split};

    #[derive(new)]
    pub struct TestEventCollector<T, V>
    where
        T: Send + Sync + 'static,
        V: Send + Sync + 'static,
    {
        info: MetricsInfo<T, V>,
    }

    impl<T, V> EventCollector for TestEventCollector<T, V>
    where
        T: Send + Sync + 'static,
        V: Send + Sync + 'static,
    {
        type ItemTrain = T;
        type ItemValid = V;

        fn on_event_train(&mut self, event: Event<Self::ItemTrain>) {
            match event {
                Event::ProcessedItem(item) => {
                    let metadata = (&item).into();
                    self.info.update_train(&item, &metadata);
                }
                Event::EndEpoch(epoch) => self.info.end_epoch_train(epoch),
            }
        }

        fn on_event_valid(&mut self, event: Event<Self::ItemValid>) {
            match event {
                Event::ProcessedItem(item) => {
                    let metadata = (&item).into();
                    self.info.update_valid(&item, &metadata);
                }
                Event::EndEpoch(epoch) => self.info.end_epoch_valid(epoch),
            }
        }

        fn find_epoch(
            &mut self,
            name: &str,
            aggregate: Aggregate,
            direction: Direction,
            split: Split,
        ) -> Option<usize> {
            self.info.find_epoch(name, aggregate, direction, split)
        }
    }
}
