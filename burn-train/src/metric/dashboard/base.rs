use crate::{
    logger::MetricLogger,
    metric::{Adaptor, Metric, MetricEntry, MetricMetadata, Numeric},
    LearnerCallback, LearnerItem,
};
use burn_core::data::dataloader::Progress;

/// Training progress.
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

/// A dashboard container for all metrics.
pub struct Dashboard<T, V>
where
    T: Send + Sync + 'static,
    V: Send + Sync + 'static,
{
    metrics_train: Vec<Box<dyn DashboardMetric<T>>>,
    metrics_valid: Vec<Box<dyn DashboardMetric<V>>>,
    metrics_train_numeric: Vec<Box<dyn DashboardNumericMetric<T>>>,
    metrics_valid_numeric: Vec<Box<dyn DashboardNumericMetric<V>>>,
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
    /// * `logger_train` - The training logger.
    /// * `logger_valid` - The validation logger.
    ///
    /// # Returns
    ///
    /// A new dashboard.
    pub fn new(
        renderer: Box<dyn DashboardRenderer>,
        logger_train: Box<dyn MetricLogger>,
        logger_valid: Box<dyn MetricLogger>,
    ) -> Self {
        Self {
            metrics_train: Vec::new(),
            metrics_valid: Vec::new(),
            metrics_train_numeric: Vec::new(),
            metrics_valid_numeric: Vec::new(),
            logger_train,
            logger_valid,
            renderer,
        }
    }

    /// Replace the current loggers with the provided ones.
    ///
    /// # Arguments
    ///
    /// * `logger_train` - The training logger.
    /// * `logger_valid` - The validation logger.
    pub fn replace_loggers(
        &mut self,
        logger_train: Box<dyn MetricLogger>,
        logger_valid: Box<dyn MetricLogger>,
    ) {
        self.logger_train = logger_train;
        self.logger_valid = logger_valid;
    }

    /// Registers a training metric.
    ///
    /// # Arguments
    ///
    /// * `metric` - The metric.
    pub fn register_train<M: Metric + 'static>(&mut self, metric: M)
    where
        T: Adaptor<M::Input>,
    {
        self.metrics_train
            .push(Box::new(MetricWrapper::new(metric)));
    }

    /// Registers a training numeric metric.
    ///
    /// # Arguments
    ///
    /// * `metric` - The metric.
    pub fn register_train_plot<M: Numeric + Metric + 'static>(&mut self, metric: M)
    where
        T: Adaptor<M::Input>,
    {
        self.metrics_train_numeric
            .push(Box::new(MetricWrapper::new(metric)));
    }

    /// Registers a validation metric.
    ///
    /// # Arguments
    ///
    /// * `metric` - The metric.
    pub fn register_valid<M: Metric + 'static>(&mut self, metric: M)
    where
        V: Adaptor<M::Input>,
    {
        self.metrics_valid
            .push(Box::new(MetricWrapper::new(metric)));
    }

    /// Registers a validation numeric metric.
    ///
    /// # Arguments
    ///
    /// * `metric` - The metric.
    pub fn register_valid_plot<M: Numeric + Metric + 'static>(&mut self, metric: M)
    where
        V: Adaptor<M::Input>,
    {
        self.metrics_valid_numeric
            .push(Box::new(MetricWrapper::new(metric)));
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
        for metric in self.metrics_train.iter_mut() {
            let state = metric.update(&item, &metadata);
            self.logger_train.log(&state);

            self.renderer
                .update_train(DashboardMetricState::Generic(state));
        }
        for metric in self.metrics_train_numeric.iter_mut() {
            let (state, value) = metric.update(&item, &metadata);
            self.logger_train.log(&state);

            self.renderer
                .update_train(DashboardMetricState::Numeric(state, value));
        }
        self.renderer.render_train(item.into());
    }

    fn on_valid_item(&mut self, item: LearnerItem<V>) {
        let metadata = (&item).into();
        for metric in self.metrics_valid.iter_mut() {
            let state = metric.update(&item, &metadata);
            self.logger_valid.log(&state);

            self.renderer
                .update_valid(DashboardMetricState::Generic(state));
        }
        for metric in self.metrics_valid_numeric.iter_mut() {
            let (state, value) = metric.update(&item, &metadata);
            self.logger_valid.log(&state);

            self.renderer
                .update_valid(DashboardMetricState::Numeric(state, value));
        }
        self.renderer.render_valid(item.into());
    }

    fn on_train_end_epoch(&mut self, epoch: usize) {
        for metric in self.metrics_train.iter_mut() {
            metric.clear();
        }
        for metric in self.metrics_train_numeric.iter_mut() {
            metric.clear();
        }
        self.logger_train.epoch(epoch + 1);
    }

    fn on_valid_end_epoch(&mut self, epoch: usize) {
        for metric in self.metrics_valid.iter_mut() {
            metric.clear();
        }
        for metric in self.metrics_valid_numeric.iter_mut() {
            metric.clear();
        }
        self.logger_valid.epoch(epoch + 1);
    }
}

trait DashboardNumericMetric<T>: Send + Sync {
    fn update(&mut self, item: &LearnerItem<T>, metadata: &MetricMetadata) -> (MetricEntry, f64);
    fn clear(&mut self);
}

trait DashboardMetric<T>: Send + Sync {
    fn update(&mut self, item: &LearnerItem<T>, metadata: &MetricMetadata) -> MetricEntry;
    fn clear(&mut self);
}

#[derive(new)]
struct MetricWrapper<M> {
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
