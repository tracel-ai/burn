use std::collections::HashMap;

use super::{ItemLazy, LearnerItem};
use crate::{
    metric::{
        Adaptor, Metric, MetricDefinition, MetricEntry, MetricId, MetricMetadata, Numeric,
        store::{MetricsUpdate, NumericMetricUpdate},
    },
    renderer::{EvaluationProgress, TrainingProgress},
};

pub(crate) struct MetricsTraining<T: ItemLazy, V: ItemLazy> {
    train: Vec<Box<dyn MetricUpdater<T::ItemSync>>>,
    valid: Vec<Box<dyn MetricUpdater<V::ItemSync>>>,
    train_numeric: Vec<Box<dyn NumericMetricUpdater<T::ItemSync>>>,
    valid_numeric: Vec<Box<dyn NumericMetricUpdater<V::ItemSync>>>,
    metric_definitions: HashMap<MetricId, MetricDefinition>,
}

pub(crate) struct MetricsEvaluation<T: ItemLazy> {
    test: Vec<Box<dyn MetricUpdater<T::ItemSync>>>,
    test_numeric: Vec<Box<dyn NumericMetricUpdater<T::ItemSync>>>,
    metric_definitions: HashMap<MetricId, MetricDefinition>,
}

impl<T: ItemLazy> Default for MetricsEvaluation<T> {
    fn default() -> Self {
        Self {
            test: Default::default(),
            test_numeric: Default::default(),
            metric_definitions: HashMap::default(),
        }
    }
}

impl<T: ItemLazy, V: ItemLazy> Default for MetricsTraining<T, V> {
    fn default() -> Self {
        Self {
            train: Vec::default(),
            valid: Vec::default(),
            train_numeric: Vec::default(),
            valid_numeric: Vec::default(),
            metric_definitions: HashMap::default(),
        }
    }
}

impl<T: ItemLazy> MetricsEvaluation<T> {
    /// Register a testing metric.
    pub(crate) fn register_test_metric<Me: Metric + 'static>(&mut self, metric: Me)
    where
        T::ItemSync: Adaptor<Me::Input> + 'static,
    {
        let metric = MetricWrapper::new(metric);
        self.register_definition(&metric);
        self.test.push(Box::new(metric))
    }

    /// Register a numeric testing metric.
    pub(crate) fn register_test_metric_numeric<Me: Metric + Numeric + 'static>(
        &mut self,
        metric: Me,
    ) where
        T::ItemSync: Adaptor<Me::Input> + 'static,
    {
        let metric = MetricWrapper::new(metric);
        self.register_definition(&metric);
        self.test_numeric.push(Box::new(metric))
    }

    fn register_definition<Me: Metric>(&mut self, metric: &MetricWrapper<Me>) {
        self.metric_definitions.insert(
            metric.id.clone(),
            MetricDefinition::new(metric.id.clone(), &metric.metric),
        );
    }

    /// Get metric definitions.
    pub(crate) fn metric_definitions(&mut self) -> Vec<MetricDefinition> {
        self.metric_definitions.values().cloned().collect()
    }

    /// Update the testing information from the testing item.
    pub(crate) fn update_test(
        &mut self,
        item: &LearnerItem<T::ItemSync>,
        metadata: &MetricMetadata,
    ) -> MetricsUpdate {
        let mut entries = Vec::with_capacity(self.test.len());
        let mut entries_numeric = Vec::with_capacity(self.test_numeric.len());

        for metric in self.test.iter_mut() {
            let state = metric.update(item, metadata);
            entries.push(state);
        }

        for metric in self.test_numeric.iter_mut() {
            let numeric_update = metric.update(item, metadata);
            entries_numeric.push(numeric_update);
        }

        MetricsUpdate::new(entries, entries_numeric)
    }
}

impl<T: ItemLazy, V: ItemLazy> MetricsTraining<T, V> {
    /// Register a training metric.
    pub(crate) fn register_train_metric<Me: Metric + 'static>(&mut self, metric: Me)
    where
        T::ItemSync: Adaptor<Me::Input> + 'static,
    {
        let metric = MetricWrapper::new(metric);
        self.register_definition(&metric);
        self.train.push(Box::new(metric))
    }

    /// Register a validation metric.
    pub(crate) fn register_valid_metric<Me: Metric + 'static>(&mut self, metric: Me)
    where
        V::ItemSync: Adaptor<Me::Input> + 'static,
    {
        let metric = MetricWrapper::new(metric);
        self.register_definition(&metric);
        self.valid.push(Box::new(metric))
    }

    /// Register a numeric training metric.
    pub(crate) fn register_train_metric_numeric<Me: Metric + Numeric + 'static>(
        &mut self,
        metric: Me,
    ) where
        T::ItemSync: Adaptor<Me::Input> + 'static,
    {
        let metric = MetricWrapper::new(metric);
        self.register_definition(&metric);
        self.train_numeric.push(Box::new(metric))
    }

    /// Register a numeric validation metric.
    pub(crate) fn register_valid_metric_numeric<Me>(&mut self, metric: Me)
    where
        V::ItemSync: Adaptor<Me::Input> + 'static,
        Me: Metric + Numeric + 'static,
    {
        let metric = MetricWrapper::new(metric);
        self.register_definition(&metric);
        self.valid_numeric.push(Box::new(metric))
    }

    fn register_definition<Me: Metric>(&mut self, metric: &MetricWrapper<Me>) {
        self.metric_definitions.insert(
            metric.id.clone(),
            MetricDefinition::new(metric.id.clone(), &metric.metric),
        );
    }

    /// Get metric definitions for all splits
    pub(crate) fn metric_definitions(&mut self) -> Vec<MetricDefinition> {
        self.metric_definitions.values().cloned().collect()
    }

    /// Update the training information from the training item.
    pub(crate) fn update_train(
        &mut self,
        item: &LearnerItem<T::ItemSync>,
        metadata: &MetricMetadata,
    ) -> MetricsUpdate {
        let mut entries = Vec::with_capacity(self.train.len());
        let mut entries_numeric = Vec::with_capacity(self.train_numeric.len());

        for metric in self.train.iter_mut() {
            let state = metric.update(item, metadata);
            entries.push(state);
        }

        for metric in self.train_numeric.iter_mut() {
            let numeric_update = metric.update(item, metadata);
            entries_numeric.push(numeric_update);
        }

        MetricsUpdate::new(entries, entries_numeric)
    }

    /// Update the training information from the validation item.
    pub(crate) fn update_valid(
        &mut self,
        item: &LearnerItem<V::ItemSync>,
        metadata: &MetricMetadata,
    ) -> MetricsUpdate {
        let mut entries = Vec::with_capacity(self.valid.len());
        let mut entries_numeric = Vec::with_capacity(self.valid_numeric.len());

        for metric in self.valid.iter_mut() {
            let state = metric.update(item, metadata);
            entries.push(state);
        }

        for metric in self.valid_numeric.iter_mut() {
            let numeric_update = metric.update(item, metadata);
            entries_numeric.push(numeric_update);
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

impl<T> From<&LearnerItem<T>> for EvaluationProgress {
    fn from(item: &LearnerItem<T>) -> Self {
        Self {
            progress: item.progress.clone(),
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

pub(crate) trait NumericMetricUpdater<T>: Send + Sync {
    fn update(&mut self, item: &LearnerItem<T>, metadata: &MetricMetadata) -> NumericMetricUpdate;
    fn clear(&mut self);
}

pub(crate) trait MetricUpdater<T>: Send + Sync {
    fn update(&mut self, item: &LearnerItem<T>, metadata: &MetricMetadata) -> MetricEntry;
    fn clear(&mut self);
}

pub(crate) struct MetricWrapper<M> {
    pub id: MetricId,
    pub metric: M,
}

impl<M: Metric> MetricWrapper<M> {
    pub fn new(metric: M) -> Self {
        Self {
            id: MetricId::new(metric.name()),
            metric,
        }
    }
}

impl<T, M> NumericMetricUpdater<T> for MetricWrapper<M>
where
    T: 'static,
    M: Metric + Numeric + 'static,
    T: Adaptor<M::Input>,
{
    fn update(&mut self, item: &LearnerItem<T>, metadata: &MetricMetadata) -> NumericMetricUpdate {
        let serialized_entry = self.metric.update(&item.item.adapt(), metadata);
        let update = MetricEntry::new(self.id.clone(), serialized_entry);
        let numeric = self.metric.value();
        let running = self.metric.running_value();

        NumericMetricUpdate {
            entry: update,
            numeric_entry: numeric,
            running_entry: running,
        }
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
        let serialized_entry = self.metric.update(&item.item.adapt(), metadata);
        MetricEntry::new(self.id.clone(), serialized_entry)
    }

    fn clear(&mut self) {
        self.metric.clear()
    }
}
