use std::{collections::HashMap, sync::Arc};

use super::{Aggregate, Direction, Event, EventStore, Split, aggregate::NumericMetricsAggregate};
use crate::logger::MetricLogger;

#[derive(Default)]
pub(crate) struct LogEventStore {
    loggers: Vec<Box<dyn MetricLogger>>,
    aggregate: NumericMetricsAggregate,
    epochs: HashMap<Split, usize>,
}

impl EventStore for LogEventStore {
    fn add_event(&mut self, event: Event, split: Split, tag: Option<Arc<String>>) {
        let epoch = *self.epochs.entry(split).or_insert(1);

        match event {
            Event::MetricsInit(definitions) => {
                definitions.iter().for_each(|def| {
                    self.loggers
                        .iter_mut()
                        .for_each(|logger| logger.log_metric_definition(def.clone()));
                });
            }
            Event::MetricsUpdate(update) => {
                let entries: Vec<_> = update
                    .entries
                    .iter()
                    .chain(update.entries_numeric.iter().map(|(entry, _value)| entry))
                    .cloned()
                    .collect();
                self.loggers
                    .iter_mut()
                    .for_each(|logger| logger.log(entries.clone(), epoch, split, tag.clone()));
            }
            Event::EndEpoch(summary) => {
                self.epochs.insert(split, summary.epoch_number + 1);
                self.loggers
                    .iter_mut()
                    .for_each(|logger| logger.log_epoch_summary(summary.clone()));
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
