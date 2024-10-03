use super::LearnerItem;
use crate::{
    metric::{store::MetricsUpdate, Adaptor, Metric, MetricEntry, MetricMetadata, Numeric},
    renderer::TrainingProgress,
};

pub(crate) struct Metrics<T, V> {
    train: Vec<Box<dyn MetricUpdater<T>>>,
    valid: Vec<Box<dyn MetricUpdater<V>>>,
    train_numeric: Vec<Box<dyn NumericMetricUpdater<T>>>,
    valid_numeric: Vec<Box<dyn NumericMetricUpdater<V>>>,
}

impl<T, V> Default for Metrics<T, V> {
    fn default() -> Self {
        Self {
            train: Vec::default(),
            valid: Vec::default(),
            train_numeric: Vec::default(),
            valid_numeric: Vec::default(),
        }
    }
}

impl<T, V> Metrics<T, V> {
    /// Register a training metric.
    pub(crate) fn register_train_metric<Me: Metric + 'static>(&mut self, metric: Me)
    where
        T: Adaptor<Me::Input> + 'static,
    {
        let metric = MetricWrapper::new(metric);
        self.train.push(Box::new(metric))
    }

    /// Register a validation metric.
    pub(crate) fn register_valid_metric<Me: Metric + 'static>(&mut self, metric: Me)
    where
        V: Adaptor<Me::Input> + 'static,
    {
        let metric = MetricWrapper::new(metric);
        self.valid.push(Box::new(metric))
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

    /// Register a numeric validation metric.
    pub(crate) fn register_valid_metric_numeric<Me: Metric + Numeric + 'static>(
        &mut self,
        metric: Me,
    ) where
        V: Adaptor<Me::Input> + 'static,
    {
        let metric = MetricWrapper::new(metric);
        self.valid_numeric.push(Box::new(metric))
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
            entries.push(state);
        }

        for metric in self.valid_numeric.iter_mut() {
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

    /// Signal the end of a validation epoch.
    pub(crate) fn end_epoch_valid(&mut self) {
        for metric in self.valid.iter_mut() {
            metric.clear();
        }
        for metric in self.valid_numeric.iter_mut() {
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
