use super::CheckpointingStrategy;
use crate::{
    checkpoint::CheckpointingAction, metric::Metric, Aggregate, Direction, EventCollector, Split,
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

impl<E: EventCollector> CheckpointingStrategy<E> for MetricCheckpointingStrategy {
    fn checkpointing(&mut self, epoch: usize, collector: &mut E) -> Vec<CheckpointingAction> {
        let best_epoch =
            match collector.find_epoch(&self.name, self.aggregate, self.direction, self.split) {
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
    use burn_core::tensor::{backend::Backend, ElementConversion, Tensor};

    use super::*;
    use crate::{
        info::MetricsInfo,
        logger::InMemoryMetricLogger,
        metric::{Adaptor, LossInput, LossMetric},
        test_utils::TestEventCollector,
        Event, LearnerItem, TestBackend,
    };

    #[test]
    fn always_keep_the_best_epoch() {
        let mut strategy = MetricCheckpointingStrategy::new::<LossMetric<TestBackend>>(
            Aggregate::Mean,
            Direction::Lowest,
            Split::Train,
        );
        let mut info = MetricsInfo::new();
        // Register an in memory logger.
        info.register_logger_train(InMemoryMetricLogger::default());
        // Register the loss metric.
        info.register_train_metric_numeric(LossMetric::<TestBackend>::new());

        let mut collector = TestEventCollector::<f64, f64>::new(info);

        // Two points for the first epoch. Mean 0.75
        let mut epoch = 1;
        item(&mut collector, 1.0, epoch);
        item(&mut collector, 0.5, epoch);
        end_epoch(&mut collector, epoch);

        // Should save the current record.
        assert_eq!(
            vec![CheckpointingAction::Save],
            strategy.checkpointing(epoch, &mut collector)
        );

        // Two points for the second epoch. Mean 0.4
        epoch += 1;
        item(&mut collector, 0.5, epoch);
        item(&mut collector, 0.3, epoch);
        end_epoch(&mut collector, epoch);

        // Should save the current record and delete the pervious one.
        assert_eq!(
            vec![CheckpointingAction::Delete(1), CheckpointingAction::Save],
            strategy.checkpointing(epoch, &mut collector)
        );

        // Two points for the last epoch. Mean 2.0
        epoch += 1;
        item(&mut collector, 1.0, epoch);
        item(&mut collector, 3.0, epoch);
        end_epoch(&mut collector, epoch);

        // Should not delete the previous record, since it's the best one, and should not save a
        // new one.
        assert!(strategy.checkpointing(epoch, &mut collector).is_empty());
    }

    fn item(collector: &mut TestEventCollector<f64, f64>, value: f64, epoch: usize) {
        let dummy_progress = burn_core::data::dataloader::Progress {
            items_processed: 1,
            items_total: 10,
        };
        let num_epochs = 3;
        let dummy_iteration = 1;

        collector.on_event_train(Event::ProcessedItem(LearnerItem::new(
            value,
            dummy_progress,
            epoch,
            num_epochs,
            dummy_iteration,
            None,
        )));
    }

    fn end_epoch(collector: &mut TestEventCollector<f64, f64>, epoch: usize) {
        collector.on_event_train(Event::EndEpoch(epoch));
        collector.on_event_valid(Event::EndEpoch(epoch));
    }

    impl<B: Backend> Adaptor<LossInput<B>> for f64 {
        fn adapt(&self) -> LossInput<B> {
            LossInput::new(Tensor::from_data([self.elem()]))
        }
    }
}
