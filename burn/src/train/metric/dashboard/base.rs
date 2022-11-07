use crate::{
    data::dataloader::Progress,
    train::{
        logger::MetricLogger,
        metric::{Metric, MetricStateDyn, Numeric},
        LearnerCallback, LearnerItem,
    },
};

pub struct TrainingProgress {
    pub progress: Progress,
    pub epoch: usize,
    pub epoch_total: usize,
    pub iteration: usize,
}

impl TrainingProgress {
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

pub enum DashboardMetricState {
    Generic(MetricStateDyn),
    Numeric(MetricStateDyn, f64),
}

pub trait DashboardRenderer: Send + Sync {
    fn update_train(&mut self, state: DashboardMetricState);
    fn update_valid(&mut self, state: DashboardMetricState);
    fn render_train(&mut self, item: TrainingProgress);
    fn render_valid(&mut self, item: TrainingProgress);
}

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

    pub fn register_train<M: Metric<T> + 'static>(&mut self, metric: M) {
        self.metrics_train
            .push(Box::new(MetricWrapper::new(metric)));
    }

    pub fn register_train_plot<M: Numeric + Metric<T> + 'static>(&mut self, metric: M) {
        self.metrics_train_numeric
            .push(Box::new(MetricWrapper::new(metric)));
    }
    pub fn register_valid<M: Metric<V> + 'static>(&mut self, metric: M) {
        self.metrics_valid
            .push(Box::new(MetricWrapper::new(metric)));
    }

    pub fn register_valid_plot<M: Numeric + Metric<V> + 'static>(&mut self, metric: M) {
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

impl<T, V> LearnerCallback<T, V> for Dashboard<T, V>
where
    T: Send + Sync + 'static,
    V: Send + Sync + 'static,
{
    fn on_train_item(&mut self, item: LearnerItem<T>) {
        for metric in self.metrics_train.iter_mut() {
            let state = metric.update(&item);
            self.logger_train.log(state.as_ref());

            self.renderer
                .update_train(DashboardMetricState::Generic(state));
        }
        for metric in self.metrics_train_numeric.iter_mut() {
            let (state, value) = metric.update(&item);
            self.logger_train.log(state.as_ref());

            self.renderer
                .update_train(DashboardMetricState::Numeric(state, value));
        }
        self.renderer.render_train(item.into());
    }

    fn on_valid_item(&mut self, item: LearnerItem<V>) {
        for metric in self.metrics_valid.iter_mut() {
            let state = metric.update(&item);
            self.logger_valid.log(state.as_ref());

            self.renderer
                .update_valid(DashboardMetricState::Generic(state));
        }
        for metric in self.metrics_valid_numeric.iter_mut() {
            let (state, value) = metric.update(&item);
            self.logger_valid.log(state.as_ref());

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
    fn update(&mut self, item: &LearnerItem<T>) -> (MetricStateDyn, f64);
    fn clear(&mut self);
}

trait DashboardMetric<T>: Send + Sync {
    fn update(&mut self, item: &LearnerItem<T>) -> MetricStateDyn;
    fn clear(&mut self);
}

#[derive(new)]
struct MetricWrapper<M> {
    metric: M,
}

impl<T, M> DashboardNumericMetric<T> for MetricWrapper<M>
where
    T: 'static,
    M: Metric<T> + Numeric + 'static,
{
    fn update(&mut self, item: &LearnerItem<T>) -> (MetricStateDyn, f64) {
        let update = self.metric.update(&item.item);
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
    M: Metric<T> + 'static,
{
    fn update(&mut self, item: &LearnerItem<T>) -> MetricStateDyn {
        self.metric.update(&item.item)
    }

    fn clear(&mut self) {
        self.metric.clear()
    }
}
