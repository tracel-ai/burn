use crate::{
    logger::MetricLogger,
    metric::{Adaptor, Metric, MetricEntry, MetricMetadata, Numeric},
    LearnerCallback, LearnerItem,
};
use burn_core::data::dataloader::Progress;

/// Training progress.
#[derive(Debug)]
pub struct TrainingProgress {
    /// The progress.
    pub progress: Progress,

    /// The epoch.
    pub epoch: usize,

    /// The total number of epochs.
    pub epoch_total: usize,

    /// The iteration.
    pub iteration: usize,
}

impl TrainingProgress {
    /// Creates a new empty training progress.
    pub fn none() -> Self {
        Self {
            progress: Progress {
                items_processed: 0,
                items_total: 0,
            },
            epoch: 0,
            epoch_total: 0,
            iteration: 0,
        }
    }
}

/// A dashboard metric.
#[derive(Debug)]
pub enum DashboardMetricState {
    /// A generic metric.
    Generic(MetricEntry),

    /// A numeric metric.
    Numeric(MetricEntry, f64),
}

/// Trait for rendering dashboard metrics.
pub trait DashboardRenderer: Send + Sync {
    /// Updates the training metric state.
    ///
    /// # Arguments
    ///
    /// * `state` - The metric state.
    fn update_train(&mut self, state: DashboardMetricState);

    /// Updates the validation metric state.
    ///
    /// # Arguments
    ///
    /// * `state` - The metric state.
    fn update_valid(&mut self, state: DashboardMetricState);

    /// Renders the training progress.
    ///
    /// # Arguments
    ///
    /// * `item` - The training progress.
    fn render_train(&mut self, item: TrainingProgress);

    /// Renders the validation progress.
    ///
    /// # Arguments
    ///
    /// * `item` - The validation progress.
    fn render_valid(&mut self, item: TrainingProgress);
}

/// A container for the metrics held by a dashboard.
pub(crate) struct Metrics<T, V>
where
    T: Send + Sync + 'static,
    V: Send + Sync + 'static,
{
    pub(crate) train: Vec<Box<dyn DashboardMetric<T>>>,
    pub(crate) valid: Vec<Box<dyn DashboardMetric<V>>>,
    pub(crate) train_numeric: Vec<Box<dyn DashboardNumericMetric<T>>>,
    pub(crate) valid_numeric: Vec<Box<dyn DashboardNumericMetric<V>>>,
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
}

/// Holds all metrics, metric loggers, and a dashboard renderer.
pub struct Dashboard<T, V>
where
    T: Send + Sync + 'static,
    V: Send + Sync + 'static,
{
    metrics: Metrics<T, V>,
    logger_train: Box<dyn MetricLogger>,
    logger_valid: Box<dyn MetricLogger>,
    renderer: Box<dyn DashboardRenderer>,
}

impl<T, V> Dashboard<T, V>
where
    T: Send + Sync + 'static,
    V: Send + Sync + 'static,
{
    /// Creates a new dashboard.
    ///
    /// # Arguments
    ///
    /// * `renderer` - The dashboard renderer.
    /// * `metrics` - The dashboard's metrics
    /// * `logger_train` - The training logger.
    /// * `logger_valid` - The validation logger.
    ///
    /// # Returns
    ///
    /// A new dashboard.
    pub(crate) fn new(
        renderer: Box<dyn DashboardRenderer>,
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

impl<T, V> LearnerCallback<T, V> for Dashboard<T, V>
where
    T: Send + Sync + 'static,
    V: Send + Sync + 'static,
{
    fn on_train_item(&mut self, item: LearnerItem<T>) {
        let metadata = (&item).into();
        for metric in self.metrics.train.iter_mut() {
            let state = metric.update(&item, &metadata);
            self.logger_train.log(&state);

            self.renderer
                .update_train(DashboardMetricState::Generic(state));
        }
        for metric in self.metrics.train_numeric.iter_mut() {
            let (state, value) = metric.update(&item, &metadata);
            self.logger_train.log(&state);

            self.renderer
                .update_train(DashboardMetricState::Numeric(state, value));
        }
        self.renderer.render_train(item.into());
    }

    fn on_valid_item(&mut self, item: LearnerItem<V>) {
        let metadata = (&item).into();
        for metric in self.metrics.valid.iter_mut() {
            let state = metric.update(&item, &metadata);
            self.logger_valid.log(&state);

            self.renderer
                .update_valid(DashboardMetricState::Generic(state));
        }
        for metric in self.metrics.valid_numeric.iter_mut() {
            let (state, value) = metric.update(&item, &metadata);
            self.logger_valid.log(&state);

            self.renderer
                .update_valid(DashboardMetricState::Numeric(state, value));
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

pub(crate) trait DashboardNumericMetric<T>: Send + Sync {
    fn update(&mut self, item: &LearnerItem<T>, metadata: &MetricMetadata) -> (MetricEntry, f64);
    fn clear(&mut self);
}

pub(crate) trait DashboardMetric<T>: Send + Sync {
    fn update(&mut self, item: &LearnerItem<T>, metadata: &MetricMetadata) -> MetricEntry;
    fn clear(&mut self);
}

#[derive(new)]
pub(crate) struct MetricWrapper<M> {
    metric: M,
}

impl<T, M> DashboardNumericMetric<T> for MetricWrapper<M>
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

impl<T, M> DashboardMetric<T> for MetricWrapper<M>
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
