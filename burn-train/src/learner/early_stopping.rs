use crate::metric::{
    store::{Aggregate, Direction, EventStoreClient, Split},
    Metric,
};

/// The condition that [early stopping strategies](EarlyStoppingStrategy) should follow.
pub enum StoppingCondition {
    /// When no improvement has happened since the given number of epochs.
    /// In other words, when no best epoch has been found.
    NoImprovementSince {
        /// The number of epochs allowed to get worsen before it get better.
        n_epochs: usize,
    },
}

/// A strategy that checks if the training should be stopped.
pub trait EarlyStoppingStrategy {
    /// Update its current state and returns if the training should be stopped.
    fn should_stop(&mut self, epoch: usize, store: &EventStoreClient) -> bool;
}

/// An [early stopping strategy](EarlyStoppingStrategy) based on a metrics collected
/// during training or validation.
pub struct MetricEarlyStoppingStrategy {
    condition: StoppingCondition,
    metric_name: String,
    aggregate: Aggregate,
    direction: Direction,
    split: Split,
    best_epoch: usize,
    best_value: f64,
}

impl EarlyStoppingStrategy for MetricEarlyStoppingStrategy {
    fn should_stop(&mut self, epoch: usize, store: &EventStoreClient) -> bool {
        let current_value =
            match store.find_metric(&self.metric_name, epoch, self.aggregate, self.split) {
                Some(value) => value,
                None => {
                    log::warn!("Can't find metric for early stopping.");
                    return false;
                }
            };

        let is_best = match self.direction {
            Direction::Lowest => current_value <= self.best_value,
            Direction::Highest => current_value >= self.best_value,
        };

        if is_best {
            log::info!(
                "New best epoch found {} {}: {}",
                epoch,
                self.metric_name,
                current_value
            );
            self.best_value = current_value;
            self.best_epoch = epoch;
            return false;
        }

        match self.condition {
            StoppingCondition::NoImprovementSince { n_epochs } => {
                let should_stop = epoch - self.best_epoch >= n_epochs;

                if should_stop {
                    log::info!("Stopping training loop, no improvement since epoch {}, {}: {},  current epoch {}, {}: {}", self.best_epoch, self.metric_name, self.best_value, epoch, self.metric_name, current_value);
                }

                should_stop
            }
        }
    }
}

impl MetricEarlyStoppingStrategy {
    /// Create a new [early stopping strategy](EarlyStoppingStrategy) based on a metrics collected
    /// during training or validation.
    ///
    /// # Notes
    ///
    /// The metric should be registered for early stopping to work, otherwise no data is collected.
    pub fn new<Me: Metric>(
        aggregate: Aggregate,
        direction: Direction,
        split: Split,
        condition: StoppingCondition,
    ) -> Self {
        let init_value = match direction {
            Direction::Lowest => f64::MAX,
            Direction::Highest => f64::MIN,
        };

        Self {
            metric_name: Me::NAME.to_string(),
            condition,
            aggregate,
            direction,
            split,
            best_epoch: 1,
            best_value: init_value,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::{
        logger::InMemoryMetricLogger,
        metric::{
            processor::{
                test_utils::{end_epoch, process_train},
                Metrics, MinimalEventProcessor,
            },
            store::LogEventStore,
            LossMetric,
        },
        TestBackend,
    };

    use super::*;

    #[test]
    fn metric_early_should_stop_when_metric_gets_worse() {
        let mut early_stopping = MetricEarlyStoppingStrategy::new::<LossMetric<TestBackend>>(
            Aggregate::Mean,
            Direction::Lowest,
            Split::Train,
            StoppingCondition::NoImprovementSince { n_epochs: 2 },
        );
        let mut store = LogEventStore::default();
        let mut metrics = Metrics::<f64, f64>::default();
        store.register_logger_train(InMemoryMetricLogger::default());
        metrics.register_train_metric_numeric(LossMetric::<TestBackend>::new());

        let store = Arc::new(EventStoreClient::new(store));
        let mut processor = MinimalEventProcessor::new(metrics, store.clone());

        // Two points for the first epoch. Mean 0.75
        let mut epoch = 1;
        process_train(&mut processor, 1.0, epoch);
        process_train(&mut processor, 0.5, epoch);
        end_epoch(&mut processor, epoch);

        assert!(
            !early_stopping.should_stop(epoch, &store),
            "Should not stop first epoch."
        );

        // Two points for the second epoch. Mean 0.4
        epoch += 1;
        process_train(&mut processor, 0.5, epoch);
        process_train(&mut processor, 0.3, epoch);
        end_epoch(&mut processor, epoch);

        assert!(
            !early_stopping.should_stop(epoch, &store),
            "Should not stop since it's better."
        );

        // Two points for the third epoch. Mean 2.0
        epoch += 1;
        process_train(&mut processor, 1.0, epoch);
        process_train(&mut processor, 3.0, epoch);
        end_epoch(&mut processor, epoch);

        assert!(
            !early_stopping.should_stop(epoch, &store),
            "Should not stop even if worsen."
        );

        // Two points for the last epoch. Mean 1.5
        epoch += 1;
        process_train(&mut processor, 1.0, epoch);
        process_train(&mut processor, 2.0, epoch);
        end_epoch(&mut processor, epoch);

        assert!(
            early_stopping.should_stop(epoch, &store),
            "Should stop since two following epochs are worse then the best."
        );
    }
}
