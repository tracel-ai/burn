use crate::{
    logger::MetricLogger,
    metric::{Adaptor, Metric, MetricEntry, MetricMetadata, Numeric},
    renderer::{MetricState, MetricsRenderer, TrainingProgress},
    LearnerCallback, LearnerItem,
};

/// Holds all metrics, metric loggers, and a metrics renderer.
pub struct MetricsCallback<T, V>
where
    T: Send + Sync + 'static,
    V: Send + Sync + 'static,
{
    metrics: Metrics<T, V>,
    logger_train: Box<dyn MetricLogger>,
    logger_valid: Box<dyn MetricLogger>,
    renderer: Box<dyn MetricsRenderer>,
}

impl<T, V> MetricsCallback<T, V>
where
    T: Send + Sync + 'static,
    V: Send + Sync + 'static,
{
    /// Creates a new metrics callback.
    ///
    /// # Arguments
    ///
    /// * `renderer` - The metrics renderer.
    /// * `metrics` - The metrics holder.
    /// * `logger_train` - The training logger.
    /// * `logger_valid` - The validation logger.
    ///
    /// # Returns
    ///
    /// A new metrics callback.
    pub(crate) fn new(
        renderer: Box<dyn MetricsRenderer>,
        metrics: Metrics<T, V>,
        logger_train: Box<dyn MetricLogger>,
        logger_valid: Box<dyn MetricLogger>,
    ) -> Self {
        Self {
            metrics,
            logger_train,
            logger_valid,
            renderer,
        }
    }
}

impl<T, V> LearnerCallback for MetricsCallback<T, V>
where
    T: Send + Sync + 'static,
    V: Send + Sync + 'static,
{
    type ItemTrain = T;
    type ItemValid = V;

    fn on_train_item(&mut self, item: LearnerItem<T>) {
        let metadata = (&item).into();
        for metric in self.metrics.train.iter_mut() {
            let state = metric.update(&item, &metadata);
            self.logger_train.log(&state);

            self.renderer.update_train(MetricState::Generic(state));
        }
        for metric in self.metrics.train_numeric.iter_mut() {
            let (state, value) = metric.update(&item, &metadata);
            self.logger_train.log(&state);

            self.renderer
                .update_train(MetricState::Numeric(state, value));
        }
        self.renderer.render_train(item.into());
    }

    fn on_valid_item(&mut self, item: LearnerItem<V>) {
        let metadata = (&item).into();
        for metric in self.metrics.valid.iter_mut() {
            let state = metric.update(&item, &metadata);
            self.logger_valid.log(&state);

            self.renderer.update_valid(MetricState::Generic(state));
        }
        for metric in self.metrics.valid_numeric.iter_mut() {
            let (state, value) = metric.update(&item, &metadata);
            self.logger_valid.log(&state);

            self.renderer
                .update_valid(MetricState::Numeric(state, value));
        }
        self.renderer.render_valid(item.into());
    }

    fn on_train_end_epoch(&mut self, epoch: usize) {
        for metric in self.metrics.train.iter_mut() {
            metric.clear();
        }
        for metric in self.metrics.train_numeric.iter_mut() {
            metric.clear();
        }
        self.logger_train.epoch(epoch + 1);
    }

    fn on_valid_end_epoch(&mut self, epoch: usize) {
        for metric in self.metrics.valid.iter_mut() {
            metric.clear();
        }
        for metric in self.metrics.valid_numeric.iter_mut() {
            metric.clear();
        }
        self.logger_valid.epoch(epoch + 1);
    }
}

/// A container for the metrics held by a metrics callback.
pub(crate) struct Metrics<T, V>
where
    T: Send + Sync + 'static,
    V: Send + Sync + 'static,
{
    train: Vec<Box<dyn MetricUpdater<T>>>,
    valid: Vec<Box<dyn MetricUpdater<V>>>,
    train_numeric: Vec<Box<dyn NumericMetricUpdater<T>>>,
    valid_numeric: Vec<Box<dyn NumericMetricUpdater<V>>>,
}

impl<T, V> Metrics<T, V>
where
    T: Send + Sync + 'static,
    V: Send + Sync + 'static,
{
    pub fn new() -> Self {
        Self {
            train: vec![],
            valid: vec![],
            train_numeric: vec![],
            valid_numeric: vec![],
        }
    }

    pub(crate) fn add_train<Me: Metric + 'static>(&mut self, metric: Me)
    where
        T: Adaptor<Me::Input>,
    {
        let metric = MetricWrapper::new(metric);
        self.train.push(Box::new(metric))
    }

    pub(crate) fn add_valid<Me: Metric + 'static>(&mut self, metric: Me)
    where
        V: Adaptor<Me::Input>,
    {
        let metric = MetricWrapper::new(metric);
        self.valid.push(Box::new(metric))
    }

    pub(crate) fn add_numeric_train<Me: Metric + Numeric + 'static>(&mut self, metric: Me)
    where
        T: Adaptor<Me::Input>,
    {
        let metric = MetricWrapper::new(metric);
        self.train_numeric.push(Box::new(metric))
    }

    pub(crate) fn add_numeric_valid<Me: Metric + Numeric + 'static>(&mut self, metric: Me)
    where
        V: Adaptor<Me::Input>,
    {
        let metric = MetricWrapper::new(metric);
        self.valid_numeric.push(Box::new(metric))
    }
}

impl<T> From<LearnerItem<T>> for TrainingProgress {
    fn from(item: LearnerItem<T>) -> Self {
        Self {
            progress: item.progress,
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
