use super::{Aggregate, Direction, Event, EventStore, Split, aggregate::NumericMetricsAggregate};
use crate::logger::MetricLogger;

pub(crate) struct LogEventStore {
    loggers: Vec<Box<dyn MetricLogger>>,
    aggregate: NumericMetricsAggregate,
    epoch: usize,
}

impl Default for LogEventStore {
    fn default() -> Self {
        Self {
            loggers: vec![],
            aggregate: NumericMetricsAggregate::default(),
            epoch: 1,
        }
    }
}

impl EventStore for LogEventStore {
    fn add_event(&mut self, event: Event, split: Split) {
        match event {
            Event::MetricsUpdate(update) => {
                update
                    .entries
                    .iter()
                    .chain(update.entries_numeric.iter().map(|(entry, _value)| entry))
                    .for_each(|entry| {
                        self.loggers
                            .iter_mut()
                            .for_each(|logger| logger.log(entry, self.epoch, split));
                    });
            }
            Event::EndEpoch(epoch) => {
                self.epoch = epoch + 1;
            }
        }
    }

    fn find_epoch(
        &mut self,
        name: &str,
        aggregate: Aggregate,
        direction: Direction,
        split: Split,
    ) -> Option<usize> {
        self.aggregate
            .find_epoch(name, split, aggregate, direction, &mut self.loggers)
    }

    fn find_metric(
        &mut self,
        name: &str,
        epoch: usize,
        aggregate: Aggregate,
        split: Split,
    ) -> Option<f64> {
        self.aggregate
            .aggregate(name, epoch, split, aggregate, &mut self.loggers)
    }
}

impl LogEventStore {
    /// Register a logger for metrics.
    pub(crate) fn register_logger<ML: MetricLogger + 'static>(&mut self, logger: ML) {
        self.loggers.push(Box::new(logger));
    }

    /// Returns whether any loggers are registered.
    pub(crate) fn has_loggers(&self) -> bool {
        !self.loggers.is_empty()
    }
}
