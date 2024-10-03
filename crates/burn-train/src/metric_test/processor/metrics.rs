use super::LearnerItem;
use crate::{
    metric_test::{store::MetricsUpdate, Adaptor, Metric, MetricEntry, MetricMetadata, Numeric},
    renderer_test::TrainingProgress,
};

pub(crate) struct Metrics<T> {
    train: Vec<Box<dyn MetricUpdater<T>>>,
    train_numeric: Vec<Box<dyn NumericMetricUpdater<T>>>,
}

impl<T> Default for Metrics<T> {
    fn default() -> Self {
        Self {
            train: Vec::default(),
            train_numeric: Vec::default(),
        }
    }
}

impl<T> Metrics<T> {
    /// Register a training metric.
    pub(crate) fn register_train_metric<Me: Metric + 'static>(&mut self, metric: Me)
    where
        T: Adaptor<Me::Input> + 'static,
    {
        let metric = MetricWrapper::new(metric);
        self.train.push(Box::new(metric))
    }

    /// Register a numeric training metric.
    pub(crate) fn register_train_metric_numeric<Me: Metric + Numeric + 'static>(
        &mut self,
        metric: Me,
    ) where
        T: Adaptor<Me::Input> + 'static,
    {
        let metric = MetricWrapper::new(metric);
        self.train_numeric.push(Box::new(metric))
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
            entries.push(state);
        }

        for metric in self.train_numeric.iter_mut() {
            let (state, value) = metric.update(item, metadata);
            entries_numeric.push((state, value));
        }

        MetricsUpdate::new(entries, entries_numeric)
    }

    /// Signal the end of a training epoch.
    pub(crate) fn end_epoch_train(&mut self) {
        for metric in self.train.iter_mut() {
            metric.clear();
        }
        for metric in self.train_numeric.iter_mut() {
            metric.clear();
        }
    }
}

impl<T> From<&LearnerItem<T>> for TrainingProgress {
    fn from(item: &LearnerItem<T>) -> Self {
        Self {
            progress: item.progress.clone(),
            epoch: item.epoch,
            epoch_total: item.epoch_total,
            iteration: item.iteration,
        }
    }
}

impl<T> From<&LearnerItem<T>> for MetricMetadata {
    fn from(item: &LearnerItem<T>) -> Self {
        Self {
            progress: item.progress.clone(),
            epoch: item.epoch,
            epoch_total: item.epoch_total,
            iteration: item.iteration,
            lr: item.lr,
        }
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
