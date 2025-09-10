use super::{Aggregate, Direction, Event, EventStore, Split, aggregate::NumericMetricsAggregate};
use crate::logger::MetricLogger;

#[derive(Default)]
pub(crate) struct LogEventStore {
    loggers_train: Vec<Box<dyn MetricLogger>>,
    loggers_valid: Vec<Box<dyn MetricLogger>>,
    loggers_test: Vec<Box<dyn MetricLogger>>,
    aggregate_train: NumericMetricsAggregate,
    aggregate_valid: NumericMetricsAggregate,
    aggregate_test: NumericMetricsAggregate,
}

impl EventStore for LogEventStore {
    fn add_event(&mut self, event: Event, split: Split) {
        match event {
            Event::MetricsUpdate(update) => match split {
                Split::Train => {
                    update
                        .entries
                        .iter()
                        .chain(update.entries_numeric.iter().map(|(entry, _value)| entry))
                        .for_each(|entry| {
                            self.loggers_train
                                .iter_mut()
                                .for_each(|logger| logger.log(entry));
                        });
                }
                Split::Valid => {
                    update
                        .entries
                        .iter()
                        .chain(update.entries_numeric.iter().map(|(entry, _value)| entry))
                        .for_each(|entry| {
                            self.loggers_valid
                                .iter_mut()
                                .for_each(|logger| logger.log(entry));
                        });
                }
                Split::Test => {
                    update
                        .entries
                        .iter()
                        .chain(update.entries_numeric.iter().map(|(entry, _value)| entry))
                        .for_each(|entry| {
                            self.loggers_test
                                .iter_mut()
                                .for_each(|logger| logger.log(entry));
                        });
                }
            },
            Event::EndEpoch(epoch) => match split {
                Split::Train => self
                    .loggers_train
                    .iter_mut()
                    .for_each(|logger| logger.end_epoch(epoch)),
                Split::Valid => self
                    .loggers_valid
                    .iter_mut()
                    .for_each(|logger| logger.end_epoch(epoch)),
                Split::Test => self
                    .loggers_test
                    .iter_mut()
                    .for_each(|logger| logger.end_epoch(epoch)),
            },
        }
    }

    fn find_epoch(
        &mut self,
        name: &str,
        aggregate: Aggregate,
        direction: Direction,
        split: Split,
    ) -> Option<usize> {
        match split {
            Split::Train => {
                self.aggregate_train
                    .find_epoch(name, aggregate, direction, &mut self.loggers_train)
            }
            Split::Valid => {
                self.aggregate_valid
                    .find_epoch(name, aggregate, direction, &mut self.loggers_valid)
            }
            Split::Test => {
                self.aggregate_test
                    .find_epoch(name, aggregate, direction, &mut self.loggers_test)
            }
        }
    }

    fn find_metric(
        &mut self,
        name: &str,
        epoch: usize,
        aggregate: Aggregate,
        split: Split,
    ) -> Option<f64> {
        match split {
            Split::Train => {
                self.aggregate_train
                    .aggregate(name, epoch, aggregate, &mut self.loggers_train)
            }
            Split::Valid => {
                self.aggregate_valid
                    .aggregate(name, epoch, aggregate, &mut self.loggers_valid)
            }
            Split::Test => {
                self.aggregate_test
                    .aggregate(name, epoch, aggregate, &mut self.loggers_test)
            }
        }
    }
}

impl LogEventStore {
    /// Register a logger for training metrics.
    pub(crate) fn register_logger_train<ML: MetricLogger + 'static>(&mut self, logger: ML) {
        self.loggers_train.push(Box::new(logger));
    }

    /// Register a logger for validation metrics.
    pub(crate) fn register_logger_valid<ML: MetricLogger + 'static>(&mut self, logger: ML) {
        self.loggers_valid.push(Box::new(logger));
    }
    /// Register a logger for testing metrics.
    pub(crate) fn register_logger_test<ML: MetricLogger + 'static>(&mut self, logger: ML) {
        self.loggers_test.push(Box::new(logger));
    }
}
