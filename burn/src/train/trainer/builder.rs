use super::callback::AsyncSupervisedTrainerCallback;
use super::SupervisedTrainer;
use crate::train::metric::dashboard::cli::CLIDashboardRenderer;
use crate::train::metric::dashboard::{Dashboard, DashboardRenderer};
use crate::train::metric::{Metric, Numeric};
use burn_tensor::backend::ADBackend;

pub struct SupervisedTrainerBuilder<T, V>
where
    T: Send + Sync + 'static,
    V: Send + Sync + 'static,
{
    dashboard: Dashboard<T, V>,
    num_epochs: usize,
}

impl<T, V> Default for SupervisedTrainerBuilder<T, V>
where
    T: Send + Sync + 'static,
    V: Send + Sync + 'static,
{
    fn default() -> Self {
        Self::new(Box::new(CLIDashboardRenderer::new()))
    }
}

impl<T, V> SupervisedTrainerBuilder<T, V>
where
    T: Send + Sync + 'static,
    V: Send + Sync + 'static,
{
    pub fn new(renderer: Box<dyn DashboardRenderer>) -> Self {
        Self {
            dashboard: Dashboard::new(renderer),
            num_epochs: 1,
        }
    }

    pub fn metric_train<M: Metric<T> + 'static>(mut self, metric: M) -> Self {
        self.dashboard.register_train(metric);
        self
    }

    pub fn metric_valid<M: Metric<V> + 'static>(mut self, metric: M) -> Self {
        self.dashboard.register_valid(metric);
        self
    }

    pub fn metric_train_plot<M: Metric<T> + Numeric + 'static>(mut self, metric: M) -> Self {
        self.dashboard.register_train_plot(metric);
        self
    }

    pub fn metric_valid_plot<M: Metric<V> + Numeric + 'static>(mut self, metric: M) -> Self {
        self.dashboard.register_valid_plot(metric);
        self
    }

    pub fn num_epochs(mut self, num_epochs: usize) -> Self {
        self.num_epochs = num_epochs;
        self
    }

    pub fn build<B: ADBackend>(self) -> SupervisedTrainer<B, T, V> {
        let callack = Box::new(self.dashboard);
        let callback = Box::new(AsyncSupervisedTrainerCallback::new(callack));

        SupervisedTrainer::new(callback, self.num_epochs)
    }
}
