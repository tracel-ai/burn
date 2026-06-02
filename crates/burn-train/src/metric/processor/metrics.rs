use std::collections::HashMap;

use core::any::Any;
use core::marker::PhantomData;

use super::TrainingItem;
use crate::{
    EvaluationItem,
    metric::{
        Adaptor, Metric, MetricDefinition, MetricEntry, MetricId, MetricMetadata, Numeric,
        store::{MetricsUpdate, NumericMetricUpdate},
    },
};

#[derive(Default)]
pub(crate) struct MetricsTraining {
    train: Vec<Box<dyn MetricUpdater>>,
    valid: Vec<Box<dyn MetricUpdater>>,
    train_numeric: Vec<Box<dyn NumericMetricUpdater>>,
    valid_numeric: Vec<Box<dyn NumericMetricUpdater>>,
    metric_definitions: HashMap<MetricId, MetricDefinition>,
}

#[derive(Default)]
pub(crate) struct MetricsEvaluation {
    test: Vec<Box<dyn MetricUpdater>>,
    test_numeric: Vec<Box<dyn NumericMetricUpdater>>,
    metric_definitions: HashMap<MetricId, MetricDefinition>,
}

impl MetricsEvaluation {
    /// Register a testing metric.
    pub(crate) fn register_test_metric<T, Me: Metric + 'static>(&mut self, metric: Me)
    where
        T: Adaptor<Me::Input> + 'static,
    {
        let metric = MetricWrapper::<T, _>::new(metric);
        self.register_definition(&metric);
        self.test.push(Box::new(metric))
    }

    /// Register a numeric testing metric.
    pub(crate) fn register_test_metric_numeric<T, Me: Metric + Numeric + 'static>(
        &mut self,
        metric: Me,
    ) where
        T: Adaptor<Me::Input> + 'static,
    {
        let metric = MetricWrapper::<T, _>::new(metric);
        self.register_definition(&metric);
        self.test_numeric.push(Box::new(metric))
    }

    fn register_definition<T, Me: Metric>(&mut self, metric: &MetricWrapper<T, Me>) {
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
        item: &EvaluationItem,
        metadata: &MetricMetadata,
    ) -> MetricsUpdate {
        let mut entries = Vec::with_capacity(self.test.len());
        let mut entries_numeric = Vec::with_capacity(self.test_numeric.len());

        for metric in self.test.iter_mut() {
            let state = metric.update(item.item.as_any(), metadata);
            entries.push(state);
        }

        for metric in self.test_numeric.iter_mut() {
            let numeric_update = metric.update(item.item.as_any(), metadata);
            entries_numeric.push(numeric_update);
        }

        MetricsUpdate::new(entries, entries_numeric)
    }
}

impl MetricsTraining {
    /// Register a training metric.
    pub(crate) fn register_train_metric<T, Me: Metric + 'static>(&mut self, metric: Me)
    where
        T: Adaptor<Me::Input> + 'static,
    {
        let metric = MetricWrapper::<T, _>::new(metric);
        self.register_definition(&metric);
        self.train.push(Box::new(metric))
    }

    /// Register a validation metric.
    pub(crate) fn register_valid_metric<V, Me: Metric + 'static>(&mut self, metric: Me)
    where
        V: Adaptor<Me::Input> + 'static,
    {
        let metric = MetricWrapper::<V, _>::new(metric);
        self.register_definition(&metric);
        self.valid.push(Box::new(metric))
    }

    /// Register a numeric training metric.
    pub(crate) fn register_train_metric_numeric<T, Me: Metric + Numeric + 'static>(
        &mut self,
        metric: Me,
    ) where
        T: Adaptor<Me::Input> + 'static,
    {
        let metric = MetricWrapper::<T, _>::new(metric);
        self.register_definition(&metric);
        self.train_numeric.push(Box::new(metric))
    }

    /// Register a numeric validation metric.
    pub(crate) fn register_valid_metric_numeric<V, Me>(&mut self, metric: Me)
    where
        V: Adaptor<Me::Input> + 'static,
        Me: Metric + Numeric + 'static,
    {
        let metric = MetricWrapper::<V, _>::new(metric);
        self.register_definition(&metric);
        self.valid_numeric.push(Box::new(metric))
    }

    fn register_definition<T, Me: Metric>(&mut self, metric: &MetricWrapper<T, Me>) {
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
        item: &TrainingItem,
        metadata: &MetricMetadata,
    ) -> MetricsUpdate {
        let mut entries = Vec::with_capacity(self.train.len());
        let mut entries_numeric = Vec::with_capacity(self.train_numeric.len());

        for metric in self.train.iter_mut() {
            let state = metric.update(item.item.as_any(), metadata);
            entries.push(state);
        }

        for metric in self.train_numeric.iter_mut() {
            let numeric_update = metric.update(item.item.as_any(), metadata);
            entries_numeric.push(numeric_update);
        }

        MetricsUpdate::new(entries, entries_numeric)
    }

    /// Update the training information from the validation item.
    pub(crate) fn update_valid(
        &mut self,
        item: &TrainingItem,
        metadata: &MetricMetadata,
    ) -> MetricsUpdate {
        let mut entries = Vec::with_capacity(self.valid.len());
        let mut entries_numeric = Vec::with_capacity(self.valid_numeric.len());

        for metric in self.valid.iter_mut() {
            let state = metric.update(item.item.as_any(), metadata);
            entries.push(state);
        }

        for metric in self.valid_numeric.iter_mut() {
            let numeric_update = metric.update(item.item.as_any(), metadata);
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

impl From<&TrainingItem> for MetricMetadata {
    fn from(item: &TrainingItem) -> Self {
        Self {
            progress: item.progress.clone(),
            iteration: item.iteration,
            lr: item.lr,
        }
    }
}

impl From<&EvaluationItem> for MetricMetadata {
    fn from(item: &EvaluationItem) -> Self {
        Self {
            progress: item.progress.clone(),
            iteration: item.iteration,
            lr: None,
        }
    }
}

/// Type-erased metric updater. The concrete model output type is recovered via
/// downcasting inside [`MetricWrapper`], so the metric collections and the whole
/// event pipeline stay non-generic over the output type.
pub(crate) trait NumericMetricUpdater: Send + Sync {
    fn update(&mut self, item: &dyn Any, metadata: &MetricMetadata) -> NumericMetricUpdate;
    fn clear(&mut self);
}

pub(crate) trait MetricUpdater: Send + Sync {
    fn update(&mut self, item: &dyn Any, metadata: &MetricMetadata) -> MetricEntry;
    fn clear(&mut self);
}

/// Binds a [`Metric`] to the concrete item type `T` it adapts from.
///
/// `T` is captured at registration (where the builder enforces
/// `T: Adaptor<M::Input>`), and recovered by downcasting the erased item in
/// `update`. `PhantomData<fn() -> T>` keeps the wrapper `Send + Sync` regardless
/// of whether `T` is.
pub(crate) struct MetricWrapper<T, M> {
    pub id: MetricId,
    pub metric: M,
    _item: PhantomData<fn() -> T>,
}

impl<T, M: Metric> MetricWrapper<T, M> {
    pub fn new(metric: M) -> Self {
        Self {
            id: MetricId::new(metric.name()),
            metric,
            _item: PhantomData,
        }
    }
}

impl<T, M> NumericMetricUpdater for MetricWrapper<T, M>
where
    T: Adaptor<M::Input> + 'static,
    M: Metric + Numeric + 'static,
{
    fn update(&mut self, item: &dyn Any, metadata: &MetricMetadata) -> NumericMetricUpdate {
        let item = item
            .downcast_ref::<T>()
            .expect("the erased item type matches the registered metric input");
        let serialized_entry = self.metric.update(&item.adapt(), metadata);
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

impl<T, M> MetricUpdater for MetricWrapper<T, M>
where
    T: Adaptor<M::Input> + 'static,
    M: Metric + 'static,
{
    fn update(&mut self, item: &dyn Any, metadata: &MetricMetadata) -> MetricEntry {
        let item = item
            .downcast_ref::<T>()
            .expect("the erased item type matches the registered metric input");
        let serialized_entry = self.metric.update(&item.adapt(), metadata);
        MetricEntry::new(self.id.clone(), serialized_entry)
    }

    fn clear(&mut self) {
        self.metric.clear()
    }
}
