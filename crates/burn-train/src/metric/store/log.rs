use std::collections::HashMap;

use super::{Aggregate, Direction, Event, EventStore, Split, aggregate::NumericMetricsAggregate};
use crate::logger::MetricLogger;

#[derive(Default)]
pub(crate) struct LogEventStore {
    loggers: Vec<Box<dyn MetricLogger>>,
    aggregate: NumericMetricsAggregate,
    epochs: HashMap<Split, usize>,
}

impl EventStore for LogEventStore {
    fn add_event(&mut self, event: Event, split: Split) {
        let epoch = *self.epochs.entry(split).or_insert(1);

        match event {
            Event::MetricsUpdate(update) => {
                update
                    .entries
                    .iter()
                    .chain(update.entries_numeric.iter().map(|(entry, _value)| entry))
                    .for_each(|entry| {
                        self.loggers
                            .iter_mut()
                            .for_each(|logger| logger.log(entry, epoch, split));
                    });
            }
            Event::EndEpoch(epoch) => {
                self.epochs.insert(split, epoch + 1);
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
