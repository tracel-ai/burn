use crate::metric::MetricEntry;

/// Event happening during the training/validation process.
pub enum Event {
    /// Signal that metrics have been updated.
    MetricsUpdate(MetricsUpdate),
    /// Signal the end of an epoch.
    EndEpoch(usize),
}

/// Contains all metric information.
#[derive(new, Clone)]
pub struct MetricsUpdate {
    /// Metrics information related to non-numeric metrics.
    pub entries: Vec<MetricEntry>,
    /// Metrics information related to numeric metrics.
    pub entries_numeric: Vec<(MetricEntry, f64)>,
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

#[derive(Copy, Clone)]
/// The split to use.
pub enum Split {
    /// The training split.
    Train,
    /// The validation split.
    Valid,
}

#[derive(Copy, Clone)]
/// The direction of the query.
pub enum Direction {
    /// Lower is better.
    Lowest,
    /// Higher is better.
    Highest,
}
