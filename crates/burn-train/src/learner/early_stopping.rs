use crate::metric::{
    Metric, MetricName,
    store::{Aggregate, Direction, EventStoreClient, Split},
};

/// The condition that [early stopping strategies](EarlyStoppingStrategy) should follow.
#[derive(Clone)]
pub enum StoppingCondition {
    /// When no improvement has happened since the given number of epochs.
    NoImprovementSince {
        /// The number of epochs allowed to worsen before it gets better.
        n_epochs: usize,
    },
}

/// A strategy that checks if the training should be stopped.
pub trait EarlyStoppingStrategy: Send {
    /// Update its current state and returns if the training should be stopped.
    fn should_stop(&mut self, epoch: usize, store: &EventStoreClient) -> bool;
}

/// A helper trait to provide type-erased cloning.
pub trait CloneEarlyStoppingStrategy: EarlyStoppingStrategy + Send {
    /// Clone into a boxed trait object.
    fn clone_box(&self) -> Box<dyn CloneEarlyStoppingStrategy>;
}

/// Blanket-implement `CloneEarlyStoppingStrategy` for any `T` that
/// already implements your strategy + `Clone` + `Send` + `'static`.
impl<T> CloneEarlyStoppingStrategy for T
where
    T: EarlyStoppingStrategy + Clone + Send + 'static,
{
    fn clone_box(&self) -> Box<dyn CloneEarlyStoppingStrategy> {
        Box::new(self.clone())
    }
}

/// Now you can `impl Clone` for the boxed trait object.
impl Clone for Box<dyn CloneEarlyStoppingStrategy> {
    fn clone(&self) -> Box<dyn CloneEarlyStoppingStrategy> {
        self.clone_box()
    }
}

/// An [early stopping strategy](EarlyStoppingStrategy) based on a metrics collected
/// during training or validation.
#[derive(Clone)]
pub struct MetricEarlyStoppingStrategy {
    condition: StoppingCondition,
    metric_name: MetricName,
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
            Direction::Lowest => current_value < self.best_value,
            Direction::Highest => current_value > self.best_value,
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
                    log::info!(
                        "Stopping training loop, no improvement since epoch {}, {}: {},  current \
                         epoch {}, {}: {}",
                        self.best_epoch,
                        self.metric_name,
                        self.best_value,
                        epoch,
                        self.metric_name,
                        current_value
                    );
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
        metric: &Me,
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
            metric_name: metric.name(),
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

    #[test]
    fn never_early_stop_while_it_is_improving() {
        test_early_stopping(
            1,
            &[
                (&[0.5, 0.3], false, "Should not stop first epoch"),
                (&[0.4, 0.3], false, "Should not stop when improving"),
                (&[0.3, 0.3], false, "Should not stop when improving"),
                (&[0.2, 0.3], false, "Should not stop when improving"),
            ],
        );
    }

    #[test]
    fn early_stop_when_no_improvement_since_two_epochs() {
        test_early_stopping(
            2,
            &[
                (&[1.0, 0.5], false, "Should not stop first epoch"),
                (&[0.5, 0.3], false, "Should not stop when improving"),
                (
                    &[1.0, 3.0],
                    false,
                    "Should not stop first time it gets worse",
                ),
                (
                    &[1.0, 2.0],
                    true,
                    "Should stop since two following epochs didn't improve",
                ),
            ],
        );
    }

    #[test]
    fn early_stop_when_stays_equal() {
        test_early_stopping(
            2,
            &[
                (&[0.5, 0.3], false, "Should not stop first epoch"),
                (
                    &[0.5, 0.3],
                    false,
                    "Should not stop first time it stars the same",
                ),
                (
                    &[0.5, 0.3],
                    true,
                    "Should stop since two following epochs didn't improve",
                ),
            ],
        );
    }

    fn test_early_stopping(n_epochs: usize, data: &[(&[f64], bool, &str)]) {
        let loss = LossMetric::<TestBackend>::new();
        let mut early_stopping = MetricEarlyStoppingStrategy::new(
            &loss,
            Aggregate::Mean,
            Direction::Lowest,
            Split::Train,
            StoppingCondition::NoImprovementSince { n_epochs },
        );
        let mut store = LogEventStore::default();
        let mut metrics = MetricsTraining::<f64, f64>::default();

        store.register_logger_train(InMemoryMetricLogger::default());
        metrics.register_train_metric_numeric(loss);

        let store = Arc::new(EventStoreClient::new(store));
        let mut processor = MinimalEventProcessor::new(metrics, store.clone());

        let mut epoch = 1;
        for (points, should_start, comment) in data {
            for point in points.iter() {
                process_train(&mut processor, *point, epoch);
            }
            end_epoch(&mut processor, epoch);

            assert_eq!(
                *should_start,
                early_stopping.should_stop(epoch, &store),
                "{comment}"
            );
            epoch += 1;
        }
    }
}
