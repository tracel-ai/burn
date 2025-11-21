use std::collections::HashMap;

use super::{ItemLazy, LearnerItem};
use crate::{
    metric::{
        Adaptor, Metric, MetricAttributes, MetricDefinition, MetricEntry, MetricId, MetricMetadata,
        Numeric, NumericEntry, store::MetricsUpdate,
    },
    renderer::{EvaluationProgress, TrainingProgress},
};

pub(crate) struct MetricsTraining<T: ItemLazy, V: ItemLazy> {
    train: Vec<Box<dyn MetricUpdater<T::ItemSync>>>,
    valid: Vec<Box<dyn MetricUpdater<V::ItemSync>>>,
    train_numeric: Vec<Box<dyn NumericMetricUpdater<T::ItemSync>>>,
    valid_numeric: Vec<Box<dyn NumericMetricUpdater<V::ItemSync>>>,
    metric_definitions: HashMap<MetricId, MetricDefinition>,
    best_metric_entries_train: HashMap<MetricId, Option<MetricEntry>>,
    best_metric_entries_valid: HashMap<MetricId, Option<MetricEntry>>,
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
            best_metric_entries_train: HashMap::default(),
            best_metric_entries_valid: HashMap::default(),
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
            let (state, value) = metric.update(item, metadata);
            entries_numeric.push((state, value));
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
        self.best_metric_entries_train
            .insert(metric.id.clone(), None);
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
        self.best_metric_entries_valid
            .insert(metric.id.clone(), None);
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

    /// Get the best numeric metric entries for the training split, for the current epoch.
    pub(crate) fn best_metric_entries_train(&mut self) -> Vec<MetricEntry> {
        self.best_metric_entries_train
            .values()
            .filter_map(|v| v.as_ref())
            .cloned()
            .collect()
    }

    /// Get the best metric entries for the validation split, for the current epoch
    pub(crate) fn best_metric_entries_valid(&mut self) -> Vec<MetricEntry> {
        self.best_metric_entries_valid
            .values()
            .filter_map(|v| v.as_ref())
            .cloned()
            .collect()
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
            let (state, value) = metric.update(item, metadata);
            entries_numeric.push((state, value));
        }

        let definitions = self.metric_definitions.clone();
        for (state, value) in entries_numeric.iter() {
            self.update_best_entry_train(state, value, &definitions[&state.metric_id]);
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
            let (state, value) = metric.update(item, metadata);
            entries_numeric.push((state, value));
        }

        let definitions = self.metric_definitions.clone();
        for (state, value) in entries_numeric.iter() {
            self.update_best_entry_valid(state, value, &definitions[&state.metric_id]);
        }

        MetricsUpdate::new(entries, entries_numeric)
    }

    fn update_best_entry_train(
        &mut self,
        state: &MetricEntry,
        value: &NumericEntry,
        definition: &MetricDefinition,
    ) {
        let id = &definition.metric_id;
        let higher_is_better = match definition.attributes.clone() {
            MetricAttributes::Numeric(attr) => attr.higher_is_better,
            MetricAttributes::None => {
                panic!("Expected numeric metric.")
            }
        };
        let entry = self
            .best_metric_entries_train
            .entry(id.clone())
            .or_insert(None);
        // Replace if value is None or better than the previous best.
        if entry.as_ref().is_none_or(|old| {
            let old = NumericEntry::deserialize(&old.serialized_entry.serialized).unwrap();
            value.better_than(&old, higher_is_better)
        }) {
            *entry = Some(state.clone());
        }
    }

    fn update_best_entry_valid(
        &mut self,
        state: &MetricEntry,
        value: &NumericEntry,
        definition: &MetricDefinition,
    ) {
        let id = &definition.metric_id;
        let higher_is_better = match definition.attributes.clone() {
            MetricAttributes::Numeric(attr) => attr.higher_is_better,
            MetricAttributes::None => {
                panic!("Expected numeric metric.")
            }
        };
        let entry = self
            .best_metric_entries_valid
            .entry(id.clone())
            .or_insert(None);
        // Replace if value is None or better than the previous best.
        if entry.as_ref().is_none_or(|old| {
            let old = NumericEntry::deserialize(&old.serialized_entry.serialized).unwrap();
            value.better_than(&old, higher_is_better)
        }) {
            *entry = Some(state.clone());
        }
    }

    /// Signal the end of a training epoch.
    pub(crate) fn end_epoch_train(&mut self) {
        for metric in self.train.iter_mut() {
            metric.clear();
        }
        for metric in self.train_numeric.iter_mut() {
            metric.clear();
        }
        for v in self.best_metric_entries_train.values_mut() {
            *v = None;
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
        for v in self.best_metric_entries_valid.values_mut() {
            *v = None;
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

trait NumericMetricUpdater<T>: Send + Sync {
    fn update(
        &mut self,
        item: &LearnerItem<T>,
        metadata: &MetricMetadata,
    ) -> (MetricEntry, NumericEntry);
    fn clear(&mut self);
}

trait MetricUpdater<T>: Send + Sync {
    fn update(&mut self, item: &LearnerItem<T>, metadata: &MetricMetadata) -> MetricEntry;
    fn clear(&mut self);
}

struct MetricWrapper<M> {
    id: MetricId,
    metric: M,
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
    fn update(
        &mut self,
        item: &LearnerItem<T>,
        metadata: &MetricMetadata,
    ) -> (MetricEntry, NumericEntry) {
        let serialized_entry = self.metric.update(&item.item.adapt(), metadata);
        let update = MetricEntry::new(self.id.clone(), serialized_entry);
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
        let serialized_entry = self.metric.update(&item.item.adapt(), metadata);
        MetricEntry::new(self.id.clone(), serialized_entry)
    }

    fn clear(&mut self) {
        self.metric.clear()
    }
}
