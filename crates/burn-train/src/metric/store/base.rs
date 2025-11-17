use std::sync::Arc;

use crate::metric::{MetricDefinition, MetricEntry, NumericEntry};

/// Event happening during the training/validation process.
pub enum Event {
    /// Signal the iniialization of the metrics
    MetricsInit(Vec<MetricDefinition>),
    /// Signal that metrics have been updated.
    MetricsUpdate(MetricsUpdate),
    /// Signal the end of an epoch.
    EndEpoch(EpochSummary),
}

/// Contains all metric information.
#[derive(new, Clone, Debug)]
pub struct MetricsUpdate {
    /// Metrics information related to non-numeric metrics.
    pub entries: Vec<MetricEntry>,
    /// Metrics information related to numeric metrics.
    pub entries_numeric: Vec<(MetricEntry, NumericEntry)>,
}

impl MetricsUpdate {
    /// Appends a tag to the config.
    pub fn tag(&mut self, tag: Arc<String>) {
        self.entries.iter_mut().for_each(|entry| {
            entry.tags.push(tag.clone());
        });
        self.entries_numeric.iter_mut().for_each(|(entry, _)| {
            entry.tags.push(tag.clone());
        });
    }
}

/// Summary information about a given epoch
#[derive(new, Clone, Debug)]
pub struct EpochSummary {
    /// Epoch number.
    pub epoch_number: usize,
    /// Dataset split (train, valid, test).
    pub split: Split,
    /// Contains the best MetricEntry reached during this epoch for each metric, for each split.
    pub best_metric_values: Vec<MetricEntry>,
}

/// Defines how training and validation events are collected and searched.
///
/// This trait also exposes methods that uses the collected data to compute useful information.
pub trait EventStore: Send {
    /// Collect a training/validation event.
    fn add_event(&mut self, event: Event, split: Split);

    /// Find the epoch following the given criteria from the collected data.
    fn find_epoch(
        &mut self,
        name: &str,
        aggregate: Aggregate,
        direction: Direction,
        split: Split,
    ) -> Option<usize>;

    /// Find the metric value for the current epoch following the given criteria.
    fn find_metric(
        &mut self,
        name: &str,
        epoch: usize,
        aggregate: Aggregate,
        split: Split,
    ) -> Option<f64>;
}

#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
/// How to aggregate the metric.
pub enum Aggregate {
    /// Compute the average.
    Mean,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// The split to use.
pub enum Split {
    /// The training split.
    Train,
    /// The validation split.
    Valid,
    /// The testing split.
    Test,
}

impl std::fmt::Display for Split {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Split::Train => write!(f, "train"),
            Split::Valid => write!(f, "valid"),
            Split::Test => write!(f, "test"),
        }
    }
}

#[derive(Copy, Clone)]
/// The direction of the query.
pub enum Direction {
    /// Lower is better.
    Lowest,
    /// Higher is better.
    Highest,
}
