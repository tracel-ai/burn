use super::CheckpointingStrategy;
use crate::{
    checkpoint::CheckpointingAction,
    metric::{
        store::{Aggregate, Direction, EventStoreClient, Split},
        Metric,
    },
};

/// Keep the best checkpoint based on a metric.
pub struct MetricCheckpointingStrategy {
    current: Option<usize>,
    aggregate: Aggregate,
    direction: Direction,
    split: Split,
    name: String,
}

impl MetricCheckpointingStrategy {
    /// Create a new metric strategy.
    pub fn new<M>(aggregate: Aggregate, direction: Direction, split: Split) -> Self
    where
        M: Metric,
    {
        Self {
            current: None,
            name: M::NAME.to_string(),
            aggregate,
            direction,
            split,
        }
    }
}

impl CheckpointingStrategy for MetricCheckpointingStrategy {
    fn checkpointing(
        &mut self,
        epoch: usize,
        store: &EventStoreClient,
    ) -> Vec<CheckpointingAction> {
        let best_epoch =
            match store.find_epoch(&self.name, self.aggregate, self.direction, self.split) {
                Some(epoch_best) => epoch_best,
                None => epoch,
            };

        let mut actions = Vec::new();

        if let Some(current) = self.current {
            if current != best_epoch {
                actions.push(CheckpointingAction::Delete(current));
            }
        }

        if best_epoch == epoch {
            actions.push(CheckpointingAction::Save);
        }

        self.current = Some(best_epoch);

        actions
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        logger::InMemoryMetricLogger,
        metric::{
            processor::{Event, EventProcessor, LearnerItem, Metrics, MinimalEventProcessor},
            store::LogEventStore,
            Adaptor, LossInput, LossMetric,
        },
        TestBackend,
    };
    use burn_core::tensor::{backend::Backend, ElementConversion, Tensor};
    use std::sync::Arc;

    use super::*;

    #[test]
    fn always_keep_the_best_epoch() {
        let mut store = LogEventStore::default();
        let mut strategy = MetricCheckpointingStrategy::new::<LossMetric<TestBackend>>(
            Aggregate::Mean,
            Direction::Lowest,
            Split::Train,
        );
        let mut metrics = Metrics::<f64, f64>::default();
        // Register an in memory logger.
        store.register_logger_train(InMemoryMetricLogger::default());
        // Register the loss metric.
        metrics.register_train_metric_numeric(LossMetric::<TestBackend>::new());
        let store = Arc::new(EventStoreClient::new(store));
        let mut processor = MinimalEventProcessor::new(metrics, store.clone());

        // Two points for the first epoch. Mean 0.75
        let mut epoch = 1;
        item(&mut processor, 1.0, epoch);
        item(&mut processor, 0.5, epoch);
        end_epoch(&mut processor, epoch);

        // Should save the current record.
        assert_eq!(
            vec![CheckpointingAction::Save],
            strategy.checkpointing(epoch, &store)
        );

        // Two points for the second epoch. Mean 0.4
        epoch += 1;
        item(&mut processor, 0.5, epoch);
        item(&mut processor, 0.3, epoch);
        end_epoch(&mut processor, epoch);

        // Should save the current record and delete the pervious one.
        assert_eq!(
            vec![CheckpointingAction::Delete(1), CheckpointingAction::Save],
            strategy.checkpointing(epoch, &store)
        );

        // Two points for the last epoch. Mean 2.0
        epoch += 1;
        item(&mut processor, 1.0, epoch);
        item(&mut processor, 3.0, epoch);
        end_epoch(&mut processor, epoch);

        // Should not delete the previous record, since it's the best one, and should not save a
        // new one.
        assert!(strategy.checkpointing(epoch, &store).is_empty());
    }

    fn item(processor: &mut MinimalEventProcessor<f64, f64>, value: f64, epoch: usize) {
        let dummy_progress = burn_core::data::dataloader::Progress {
            items_processed: 1,
            items_total: 10,
        };
        let num_epochs = 3;
        let dummy_iteration = 1;

        processor.add_event_train(Event::ProcessedItem(LearnerItem::new(
            value,
            dummy_progress,
            epoch,
            num_epochs,
            dummy_iteration,
            None,
        )));
    }

    fn end_epoch(processor: &mut MinimalEventProcessor<f64, f64>, epoch: usize) {
        processor.add_event_train(Event::EndEpoch(epoch));
        processor.add_event_valid(Event::EndEpoch(epoch));
    }

    impl<B: Backend> Adaptor<LossInput<B>> for f64 {
        fn adapt(&self) -> LossInput<B> {
            LossInput::new(Tensor::from_data([self.elem()]))
        }
    }
}
