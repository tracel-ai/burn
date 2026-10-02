use std::collections::HashMap;

use burn_core::tensor::TensorReadError;

use super::{ItemLazy, MetricError, TrainingItem};
use crate::{
    EvaluationItem,
    metric::{
        Adaptor, Metric, MetricDefinition, MetricEntry, MetricId, MetricMetadata, MetricName,
        Numeric,
        store::{MetricsUpdate, NumericMetricUpdate, Split},
    },
};

pub(crate) struct MetricsTraining<T: ItemLazy, V: ItemLazy> {
    train: Vec<Box<dyn MetricUpdater<T>>>,
    valid: Vec<Box<dyn MetricUpdater<V>>>,
    train_numeric: Vec<Box<dyn NumericMetricUpdater<T>>>,
    valid_numeric: Vec<Box<dyn NumericMetricUpdater<V>>>,
    // Vec preserves metrics registration order; reflected in TUI tabs order
    metric_definitions: Vec<MetricDefinition>,
}

pub(crate) struct MetricsEvaluation<T: ItemLazy> {
    test: Vec<Box<dyn MetricUpdater<T>>>,
    test_numeric: Vec<Box<dyn NumericMetricUpdater<T>>>,
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
            metric_definitions: Vec::default(),
        }
    }
}

impl<T: ItemLazy> MetricsEvaluation<T> {
    /// Register a testing metric.
    pub(crate) fn register_test_metric<Me: Metric + 'static>(&mut self, metric: Me)
    where
        T: Adaptor<Me::Input> + 'static,
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
        T: Adaptor<Me::Input> + 'static,
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
    ///
    /// Every metric is updated, even after one fails: returns the entries of those that
    /// succeeded and the failures of the others.
    pub(crate) fn update_test(
        &mut self,
        item: &EvaluationItem<T>,
        metadata: &MetricMetadata,
        split: Split,
    ) -> (MetricsUpdate, Vec<MetricError>) {
        update_metrics(
            &mut self.test,
            &mut self.test_numeric,
            &item.item,
            metadata,
            split,
        )
    }
}

impl<T: ItemLazy, V: ItemLazy> MetricsTraining<T, V> {
    /// Register a training metric.
    pub(crate) fn register_train_metric<Me: Metric + 'static>(&mut self, metric: Me)
    where
        T: Adaptor<Me::Input> + 'static,
    {
        let metric = MetricWrapper::new(metric);
        self.register_definition(&metric);
        self.train.push(Box::new(metric))
    }

    /// Register a validation metric.
    pub(crate) fn register_valid_metric<Me: Metric + 'static>(&mut self, metric: Me)
    where
        V: Adaptor<Me::Input> + 'static,
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
        T: Adaptor<Me::Input> + 'static,
    {
        let metric = MetricWrapper::new(metric);
        self.register_definition(&metric);
        self.train_numeric.push(Box::new(metric))
    }

    /// Register a numeric validation metric.
    pub(crate) fn register_valid_metric_numeric<Me>(&mut self, metric: Me)
    where
        V: Adaptor<Me::Input> + 'static,
        Me: Metric + Numeric + 'static,
    {
        let metric = MetricWrapper::new(metric);
        self.register_definition(&metric);
        self.valid_numeric.push(Box::new(metric))
    }

    fn register_definition<Me: Metric>(&mut self, metric: &MetricWrapper<Me>) {
        // Avoid duplicate definitions if the same metric is registered for both train and valid
        if !self
            .metric_definitions
            .iter()
            .any(|def| def.metric_id == metric.id)
        {
            self.metric_definitions
                .push(MetricDefinition::new(metric.id.clone(), &metric.metric));
        }
    }

    /// Get metric definitions for all splits
    pub(crate) fn metric_definitions(&mut self) -> Vec<MetricDefinition> {
        self.metric_definitions.clone()
    }

    /// Update the training information from the training item.
    pub(crate) fn update_train(
        &mut self,
        item: &TrainingItem<T>,
        metadata: &MetricMetadata,
    ) -> (MetricsUpdate, Vec<MetricError>) {
        update_metrics(
            &mut self.train,
            &mut self.train_numeric,
            &item.item,
            metadata,
            Split::Train,
        )
    }

    /// Update the training information from the validation item.
    pub(crate) fn update_valid(
        &mut self,
        item: &TrainingItem<V>,
        metadata: &MetricMetadata,
    ) -> (MetricsUpdate, Vec<MetricError>) {
        update_metrics(
            &mut self.valid,
            &mut self.valid_numeric,
            &item.item,
            metadata,
            Split::Valid,
        )
    }

    /// Signal the end of a training epoch.
    /// Returns the final metric entries for the epoch.
    ///
    /// Every metric is computed and cleared, even after one fails, so the next epoch starts
    /// clean: returns the entries of those that succeeded and the failures of the others.
    pub(crate) fn end_epoch_train(&mut self) -> (MetricsUpdate, Vec<MetricError>) {
        end_epoch_metrics(&mut self.train, &mut self.train_numeric, Split::Train)
    }

    /// Signal the end of a validation epoch.
    /// Returns the final metric entries for the epoch.
    ///
    /// Every metric is computed and cleared, even after one fails, so the next epoch starts
    /// clean: returns the entries of those that succeeded and the failures of the others.
    pub(crate) fn end_epoch_valid(&mut self) -> (MetricsUpdate, Vec<MetricError>) {
        end_epoch_metrics(&mut self.valid, &mut self.valid_numeric, Split::Valid)
    }
}

pub(crate) fn update_metrics<I>(
    metrics: &mut [Box<dyn MetricUpdater<I>>],
    metrics_numeric: &mut [Box<dyn NumericMetricUpdater<I>>],
    item: &I,
    metadata: &MetricMetadata,
    split: Split,
) -> (MetricsUpdate, Vec<MetricError>) {
    let mut entries = Vec::with_capacity(metrics.len());
    let mut entries_numeric = Vec::with_capacity(metrics_numeric.len());
    let mut failures = Vec::new();
    let mut fail = |metric: MetricName, source| {
        failures.push(MetricError {
            metric,
            split: split.clone(),
            source,
        })
    };

    for metric in metrics.iter_mut() {
        match metric.update(item, metadata) {
            Ok(entry) => entries.push(entry),
            Err(source) => fail(metric.name(), source),
        }
    }
    for metric in metrics_numeric.iter_mut() {
        match metric.update(item, metadata) {
            Ok(entry) => entries_numeric.push(entry),
            Err(source) => fail(metric.name(), source),
        }
    }

    (MetricsUpdate::new(entries, entries_numeric), failures)
}

pub(crate) fn end_epoch_metrics<I>(
    metrics: &mut [Box<dyn MetricUpdater<I>>],
    metrics_numeric: &mut [Box<dyn NumericMetricUpdater<I>>],
    split: Split,
) -> (MetricsUpdate, Vec<MetricError>) {
    let mut entries = Vec::with_capacity(metrics.len());
    let mut entries_numeric = Vec::with_capacity(metrics_numeric.len());
    let mut failures = Vec::new();
    let mut fail = |metric: MetricName, source| {
        failures.push(MetricError {
            metric,
            split: split.clone(),
            source,
        })
    };

    for metric in metrics.iter_mut() {
        match metric.compute() {
            Ok(entry) => entries.push(entry),
            Err(source) => fail(metric.name(), source),
        }
        metric.clear();
    }
    for metric in metrics_numeric.iter_mut() {
        match metric.compute() {
            Ok(entry) => entries_numeric.push(entry),
            Err(source) => fail(metric.name(), source),
        }
        metric.clear();
    }

    (MetricsUpdate::new(entries, entries_numeric), failures)
}

impl<T> From<&TrainingItem<T>> for MetricMetadata {
    fn from(item: &TrainingItem<T>) -> Self {
        Self {
            progress: item.progress.clone(),
            iteration: item.iteration,
            lr: item.lr.clone(),
        }
    }
}

impl<T> From<&EvaluationItem<T>> for MetricMetadata {
    fn from(item: &EvaluationItem<T>) -> Self {
        Self {
            progress: item.progress.clone(),
            iteration: item.iteration,
            lr: None,
        }
    }
}

pub(crate) trait NumericMetricUpdater<T>: Send + Sync {
    fn update(
        &mut self,
        item: &T,
        metadata: &MetricMetadata,
    ) -> Result<NumericMetricUpdate, TensorReadError>;
    fn compute(&mut self) -> Result<NumericMetricUpdate, TensorReadError>;
    fn clear(&mut self);
    fn name(&self) -> MetricName;
}

pub(crate) trait MetricUpdater<T>: Send + Sync {
    fn update(
        &mut self,
        item: &T,
        metadata: &MetricMetadata,
    ) -> Result<MetricEntry, TensorReadError>;
    fn compute(&mut self) -> Result<MetricEntry, TensorReadError>;
    fn clear(&mut self);
    fn name(&self) -> MetricName;
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
    fn update(
        &mut self,
        item: &T,
        metadata: &MetricMetadata,
    ) -> Result<NumericMetricUpdate, TensorReadError> {
        let serialized_entry = self.metric.update(&item.adapt(), metadata)?;
        let update = MetricEntry::new(self.id.clone(), serialized_entry);
        let numeric = self.metric.value();
        let running = self.metric.running_value();

        Ok(NumericMetricUpdate {
            entry: update,
            numeric_entry: numeric,
            running_entry: running,
        })
    }

    fn compute(&mut self) -> Result<NumericMetricUpdate, TensorReadError> {
        let serialized_entry = self.metric.compute()?;
        let update = MetricEntry::new(self.id.clone(), serialized_entry);
        let final_entry = self.metric.final_value();

        Ok(NumericMetricUpdate {
            entry: update,
            // Running entry is not applicable. This is the final epoch-level value computed.
            numeric_entry: Some(final_entry),
            running_entry: None,
        })
    }

    fn clear(&mut self) {
        self.metric.clear()
    }

    fn name(&self) -> MetricName {
        self.metric.name()
    }
}

impl<T, M> MetricUpdater<T> for MetricWrapper<M>
where
    T: 'static,
    M: Metric + 'static,
    T: Adaptor<M::Input>,
{
    fn update(
        &mut self,
        item: &T,
        metadata: &MetricMetadata,
    ) -> Result<MetricEntry, TensorReadError> {
        let serialized_entry = self.metric.update(&item.adapt(), metadata)?;
        Ok(MetricEntry::new(self.id.clone(), serialized_entry))
    }

    fn compute(&mut self) -> Result<MetricEntry, TensorReadError> {
        let serialized_entry = self.metric.compute()?;
        Ok(MetricEntry::new(self.id.clone(), serialized_entry))
    }

    fn clear(&mut self) {
        self.metric.clear()
    }

    fn name(&self) -> MetricName {
        self.metric.name()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::{SerializedEntry, processor::EventProcessorError};
    use burn_std::ExecutionError;
    use std::sync::Arc;

    /// A metric that fails its update, its compute, or neither, and counts its clears.
    #[derive(Clone)]
    struct TestMetric {
        name: &'static str,
        fail_update: bool,
        fail_compute: bool,
        cleared: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl TestMetric {
        fn new(name: &'static str, fail_update: bool, fail_compute: bool) -> Self {
            Self {
                name,
                fail_update,
                fail_compute,
                cleared: Default::default(),
            }
        }

        fn failure(&self) -> TensorReadError {
            ExecutionError::with_context(format!("{} could not read", self.name)).into()
        }
    }

    impl Metric for TestMetric {
        type Input = ();

        fn name(&self) -> MetricName {
            Arc::new(self.name.to_string())
        }

        fn update(
            &mut self,
            _item: &(),
            _metadata: &MetricMetadata,
        ) -> Result<SerializedEntry, TensorReadError> {
            match self.fail_update {
                true => Err(self.failure()),
                false => Ok(SerializedEntry::new(self.name.into(), self.name.into())),
            }
        }

        fn compute(&mut self) -> Result<SerializedEntry, TensorReadError> {
            match self.fail_compute {
                true => Err(self.failure()),
                false => Ok(SerializedEntry::new(self.name.into(), self.name.into())),
            }
        }

        fn clear(&mut self) {
            self.cleared
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }

    fn metrics(list: &[TestMetric]) -> Vec<Box<dyn MetricUpdater<()>>> {
        list.iter()
            .map(|metric| Box::new(MetricWrapper::new(metric.clone())) as Box<_>)
            .collect()
    }

    fn names(failures: &[MetricError]) -> Vec<String> {
        failures.iter().map(|err| err.to_string()).collect()
    }

    fn metadata() -> MetricMetadata {
        MetricMetadata {
            progress: burn_core::data::dataloader::Progress {
                items_processed: 1,
                items_total: 1,
                unit: None,
            },
            iteration: None,
            lr: None,
        }
    }

    #[test]
    fn update_reports_failing_metric_and_keeps_others() {
        let list = [
            TestMetric::new("first", true, false),
            TestMetric::new("working", false, false),
            TestMetric::new("last", true, false),
        ];

        let (update, failures) =
            update_metrics(&mut metrics(&list), &mut [], &(), &metadata(), Split::Train);

        assert_eq!(
            update.entries.len(),
            1,
            "the working metric's entry is kept"
        );
        assert_eq!(
            names(&failures),
            [
                "train/first: An error happened during execution\nCaused by:\n  first could not read",
                "train/last: An error happened during execution\nCaused by:\n  last could not read",
            ]
        );
    }

    #[test]
    fn epoch_end_clears_metrics_and_reports_failures() {
        let list = [
            TestMetric::new("first", false, true),
            TestMetric::new("working", false, false),
            TestMetric::new("last", false, true),
        ];

        let (update, failures) = end_epoch_metrics(&mut metrics(&list), &mut [], Split::Valid);

        assert_eq!(
            update.entries.len(),
            1,
            "the working metric's entry is kept"
        );
        let failed: Vec<_> = failures.iter().map(|err| err.metric.as_str()).collect();
        assert_eq!(failed, ["first", "last"]);
        assert!(failures.iter().all(|err| err.split == Split::Valid));
        for metric in &list {
            assert_eq!(
                metric.cleared.load(std::sync::atomic::Ordering::SeqCst),
                1,
                "{} must be cleared so the next epoch starts clean",
                metric.name
            );
        }
    }

    #[test]
    fn metrics_error_reports_failures_and_poisoned_device() {
        let failure = |metric: &str, source: ExecutionError| MetricError {
            metric: Arc::new(metric.to_string()),
            split: Split::Test(Some(Arc::new("holdout".to_string()))),
            source: source.into(),
        };
        let error = EventProcessorError::from_errors(vec![
            failure("Accuracy", ExecutionError::with_context("refused")),
            failure("Loss", ExecutionError::device_poisoned("status 700")),
        ])
        .unwrap_err();

        assert!(error.is_device_poisoned());
        let message = error.to_string();
        assert!(
            message.starts_with("Event processing failed 2 times:"),
            "{message}"
        );
        assert!(message.contains("test/holdout/Accuracy"), "{message}");
        assert!(message.contains("test/holdout/Loss"), "{message}");

        assert!(EventProcessorError::from_errors(Vec::new()).is_ok());
    }

    #[test]
    fn a_sync_failure_is_one_failure_naming_its_split() {
        let error = EventProcessorError::sync(
            Split::Test(Some(Arc::new("holdout".to_string()))),
            ExecutionError::device_poisoned("status 700"),
        );

        assert!(error.is_device_poisoned());
        assert_eq!(
            error.failures().len(),
            1,
            "one failure, whatever the metrics"
        );
        let message = error.to_string();
        assert!(
            message.contains("test/holdout: the event could not be synced"),
            "{message}"
        );
    }
}
