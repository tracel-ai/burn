use super::NumericMetricsAggregate;
use crate::{
    logger::MetricLogger,
    metric::{Adaptor, Metric, MetricEntry, MetricMetadata, Numeric},
    Aggregate, Direction, LearnerItem, Split,
};

/// Metrics information collected during training.
pub struct MetricsInfo<T, V>
where
    T: Send + Sync + 'static,
    V: Send + Sync + 'static,
{
    train: Vec<Box<dyn MetricUpdater<T>>>,
    valid: Vec<Box<dyn MetricUpdater<V>>>,
    train_numeric: Vec<Box<dyn NumericMetricUpdater<T>>>,
    valid_numeric: Vec<Box<dyn NumericMetricUpdater<V>>>,
    loggers_train: Vec<Box<dyn MetricLogger>>,
    loggers_valid: Vec<Box<dyn MetricLogger>>,
    aggregate_train: NumericMetricsAggregate,
    aggregate_valid: NumericMetricsAggregate,
}

#[derive(new)]
pub(crate) struct MetricsUpdate {
    pub(crate) entries: Vec<MetricEntry>,
    pub(crate) entries_numeric: Vec<(MetricEntry, f64)>,
}

impl<T, V> MetricsInfo<T, V>
where
    T: Send + Sync + 'static,
    V: Send + Sync + 'static,
{
    pub(crate) fn new() -> Self {
        Self {
            train: vec![],
            valid: vec![],
            train_numeric: vec![],
            valid_numeric: vec![],
            loggers_train: vec![],
            loggers_valid: vec![],
            aggregate_train: NumericMetricsAggregate::default(),
            aggregate_valid: NumericMetricsAggregate::default(),
        }
    }

    /// Signal the end of a training epoch.
    pub(crate) fn end_epoch_train(&mut self, epoch: usize) {
        for metric in self.train.iter_mut() {
            metric.clear();
        }
        for metric in self.train_numeric.iter_mut() {
            metric.clear();
        }
        for logger in self.loggers_train.iter_mut() {
            logger.epoch(epoch + 1);
        }
    }

    /// Signal the end of a validation epoch.
    pub(crate) fn end_epoch_valid(&mut self, epoch: usize) {
        for metric in self.valid.iter_mut() {
            metric.clear();
        }
        for metric in self.valid_numeric.iter_mut() {
            metric.clear();
        }
        for logger in self.loggers_valid.iter_mut() {
            logger.epoch(epoch + 1);
        }
    }

    /// Update the training information from the training item.
    pub(crate) fn update_train(
        &mut self,
        item: &LearnerItem<T>,
        metadata: &MetricMetadata,
    ) -> MetricsUpdate {
        let mut entries = Vec::with_capacity(self.train.len());
        let mut entries_numeric = Vec::with_capacity(self.train_numeric.len());

        for metric in self.train.iter_mut() {
            let state = metric.update(item, metadata);

            for logger in self.loggers_train.iter_mut() {
                logger.log(&state);
            }

            entries.push(state);
        }

        for metric in self.train_numeric.iter_mut() {
            let (state, value) = metric.update(item, metadata);
            for logger in self.loggers_train.iter_mut() {
                logger.log(&state);
            }

            entries_numeric.push((state, value));
        }

        MetricsUpdate::new(entries, entries_numeric)
    }

    /// Update the training information from the validation item.
    pub(crate) fn update_valid(
        &mut self,
        item: &LearnerItem<V>,
        metadata: &MetricMetadata,
    ) -> MetricsUpdate {
        let mut entries = Vec::with_capacity(self.valid.len());
        let mut entries_numeric = Vec::with_capacity(self.valid_numeric.len());

        for metric in self.valid.iter_mut() {
            let state = metric.update(item, metadata);

            for logger in self.loggers_valid.iter_mut() {
                logger.log(&state);
            }

            entries.push(state);
        }

        for metric in self.valid_numeric.iter_mut() {
            let (state, value) = metric.update(item, metadata);
            for logger in self.loggers_valid.iter_mut() {
                logger.log(&state);
            }

            entries_numeric.push((state, value));
        }

        MetricsUpdate::new(entries, entries_numeric)
    }

    /// Find the epoch corresponding to the given criteria.
    pub(crate) fn find_epoch(
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
        }
    }

    /// Register a logger for training metrics.
    pub(crate) fn register_logger_train<ML: MetricLogger + 'static>(&mut self, logger: ML) {
        self.loggers_train.push(Box::new(logger));
    }

    /// Register a logger for validation metrics.
    pub(crate) fn register_logger_valid<ML: MetricLogger + 'static>(&mut self, logger: ML) {
        self.loggers_valid.push(Box::new(logger));
    }

    /// Register a training metric.
    pub(crate) fn register_metric_train<Me: Metric + 'static>(&mut self, metric: Me)
    where
        T: Adaptor<Me::Input>,
    {
        let metric = MetricWrapper::new(metric);
        self.train.push(Box::new(metric))
    }

    /// Register a validation metric.
    pub(crate) fn register_valid_metric<Me: Metric + 'static>(&mut self, metric: Me)
    where
        V: Adaptor<Me::Input>,
    {
        let metric = MetricWrapper::new(metric);
        self.valid.push(Box::new(metric))
    }

    /// Register a numeric training metric.
    pub(crate) fn register_train_metric_numeric<Me: Metric + Numeric + 'static>(
        &mut self,
        metric: Me,
    ) where
        T: Adaptor<Me::Input>,
    {
        let metric = MetricWrapper::new(metric);
        self.train_numeric.push(Box::new(metric))
    }

    /// Register a numeric validation metric.
    pub(crate) fn register_valid_metric_numeric<Me: Metric + Numeric + 'static>(
        &mut self,
        metric: Me,
    ) where
        V: Adaptor<Me::Input>,
    {
        let metric = MetricWrapper::new(metric);
        self.valid_numeric.push(Box::new(metric))
    }
}

trait NumericMetricUpdater<T>: Send + Sync {
    fn update(&mut self, item: &LearnerItem<T>, metadata: &MetricMetadata) -> (MetricEntry, f64);
    fn clear(&mut self);
}

trait MetricUpdater<T>: Send + Sync {
    fn update(&mut self, item: &LearnerItem<T>, metadata: &MetricMetadata) -> MetricEntry;
    fn clear(&mut self);
}

#[derive(new)]
struct MetricWrapper<M> {
    metric: M,
}

impl<T, M> NumericMetricUpdater<T> for MetricWrapper<M>
where
    T: 'static,
    M: Metric + Numeric + 'static,
    T: Adaptor<M::Input>,
{
    fn update(&mut self, item: &LearnerItem<T>, metadata: &MetricMetadata) -> (MetricEntry, f64) {
        let update = self.metric.update(&item.item.adapt(), metadata);
        let numeric = self.metric.value();

        (update, numeric)
    }

    fn clear(&mut self) {
        self.metric.clear()
    }
}

impl<T, M> MetricUpdater<T> for MetricWrapper<M>
where
    T: 'static,
    M: Metric + 'static,
    T: Adaptor<M::Input>,
{
    fn update(&mut self, item: &LearnerItem<T>, metadata: &MetricMetadata) -> MetricEntry {
        self.metric.update(&item.item.adapt(), metadata)
    }

    fn clear(&mut self) {
        self.metric.clear()
    }
}
