use super::CheckpointingStrategy;
use crate::{
    checkpoint::CheckpointingAction,
    metric::{
        Metric, MetricName,
        store::{Aggregate, Direction, EventStoreClient, Split},
    },
};

/// Keep the best checkpoint based on a metric.
pub struct MetricCheckpointingStrategy {
    current: Option<usize>,
    aggregate: Aggregate,
    direction: Direction,
    split: Split,
    name: MetricName,
}

impl MetricCheckpointingStrategy {
    /// Create a new metric checkpointing strategy.
    pub fn new<M>(metric: &M, aggregate: Aggregate, direction: Direction, split: Split) -> Self
    where
        M: Metric,
    {
        Self {
            current: None,
            name: metric.name(),
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

        if let Some(current) = self.current
            && current != best_epoch
        {
            actions.push(CheckpointingAction::Delete(current));
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
        TestBackend,
        logger::InMemoryMetricLogger,
        metric::{
            LossMetric,
            processor::{
                MetricsTraining, MinimalEventProcessor,
                test_utils::{end_epoch, process_train},
            },
            store::LogEventStore,
        },
    };

    use super::*;
    use std::sync::Arc;

    #[test]
    fn always_keep_the_best_epoch() {
        let loss = LossMetric::<TestBackend>::new();
        let mut store = LogEventStore::default();
        let mut strategy = MetricCheckpointingStrategy::new(
            &loss,
            Aggregate::Mean,
            Direction::Lowest,
            Split::Train,
        );
        let mut metrics = MetricsTraining::<f64, f64>::default();
        // Register an in memory logger.
        store.register_logger_train(InMemoryMetricLogger::default());
        // Register the loss metric.
        metrics.register_train_metric_numeric(loss);
        let store = Arc::new(EventStoreClient::new(store));
        let mut processor = MinimalEventProcessor::new(metrics, store.clone());

        // Two points for the first epoch. Mean 0.75
        let mut epoch = 1;
        process_train(&mut processor, 1.0, epoch);
        process_train(&mut processor, 0.5, epoch);
        end_epoch(&mut processor, epoch);

        // Should save the current record.
        assert_eq!(
            vec![CheckpointingAction::Save],
            strategy.checkpointing(epoch, &store)
        );

        // Two points for the second epoch. Mean 0.4
        epoch += 1;
        process_train(&mut processor, 0.5, epoch);
        process_train(&mut processor, 0.3, epoch);
        end_epoch(&mut processor, epoch);

        // Should save the current record and delete the previous one.
        assert_eq!(
            vec![CheckpointingAction::Delete(1), CheckpointingAction::Save],
            strategy.checkpointing(epoch, &store)
        );

        // Two points for the last epoch. Mean 2.0
        epoch += 1;
        process_train(&mut processor, 1.0, epoch);
        process_train(&mut processor, 3.0, epoch);
        end_epoch(&mut processor, epoch);

        // Should not delete the previous record, since it's the best one, and should not save a
        // new one.
        assert!(strategy.checkpointing(epoch, &store).is_empty());
    }
}
