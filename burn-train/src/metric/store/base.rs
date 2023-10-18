use crate::metric::MetricEntry;

/// Event happening during the training/validation process.
pub enum Event {
    /// Signal that an item have been processed.
    MetricsUpdate(MetricsUpdate),
    /// Signal the end of an epoch.
    EndEpoch(usize),
}

/// Contains all metric information.
#[derive(new)]
pub struct MetricsUpdate {
    /// Metric information related to non-numeric metrics.
    pub entries: Vec<MetricEntry>,
    /// Metric information related to numeric metrics.
    pub entries_numeric: Vec<(MetricEntry, f64)>,
}

/// Defines how training and validation events are collected and searched.
///
/// This trait also exposes methods that uses the collected data to compute useful information.
pub(crate) trait EventStore: Send {
    /// Collect the training event.
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

#[derive(Copy, Clone)]
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
