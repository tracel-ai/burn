use super::Learner;
use crate::module::ADModule;
use crate::train::checkpoint::{AsyncCheckpointer, Checkpointer, FileCheckpointer};
use crate::train::metric::dashboard::cli::CLIDashboardRenderer;
use crate::train::metric::dashboard::{Dashboard, DashboardRenderer};
use crate::train::metric::{Metric, Numeric};
use crate::train::AsyncTrainerCallback;
use burn_tensor::backend::ADBackend;
use burn_tensor::Element;
use std::sync::Arc;

pub struct LearnerBuilder<B, T, V>
where
    T: Send + Sync + 'static,
    V: Send + Sync + 'static,
    B: ADBackend,
{
    dashboard: Dashboard<T, V>,
    checkpointer_model: Option<Arc<dyn Checkpointer<B::Elem> + Send + Sync>>,
    checkpointer_optimizer: Option<Arc<dyn Checkpointer<B::Elem> + Send + Sync>>,
    num_epochs: usize,
    checkpoint: Option<usize>,
}

impl<B, T, V> Default for LearnerBuilder<B, T, V>
where
    T: Send + Sync + 'static,
    V: Send + Sync + 'static,
    B: ADBackend,
{
    fn default() -> Self {
        Self::new(Box::new(CLIDashboardRenderer::new()))
    }
}

impl<B, T, V> LearnerBuilder<B, T, V>
where
    T: Send + Sync + 'static,
    V: Send + Sync + 'static,
    B: ADBackend,
{
    pub fn new(renderer: Box<dyn DashboardRenderer>) -> Self {
        Self {
            dashboard: Dashboard::new(renderer),
            num_epochs: 1,
            checkpoint: None,
            checkpointer_model: None,
            checkpointer_optimizer: None,
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

    pub fn checkpoint(mut self, checkpoint: usize) -> Self {
        self.checkpoint = Some(checkpoint);
        self
    }

    pub fn with_file_checkpointer<P: Element + serde::de::DeserializeOwned + serde::Serialize>(
        mut self,
        directory: &str,
    ) -> Self {
        self.checkpointer_model = Some(Arc::new(FileCheckpointer::<P>::new(directory, "model")));
        self.checkpointer_optimizer =
            Some(Arc::new(FileCheckpointer::<P>::new(directory, "optim")));
        self
    }

    pub fn build<M, O>(self, model: M, optim: O) -> Learner<M, O, T, V>
    where
        M: ADModule<ADBackend = B>,
    {
        let callack = Box::new(self.dashboard);
        let callback = Box::new(AsyncTrainerCallback::new(callack));

        let create_checkpointer = |checkpointer| match checkpointer {
            Some(checkpointer) => {
                let checkpointer: Box<dyn Checkpointer<B::Elem>> =
                    Box::new(AsyncCheckpointer::new(checkpointer));
                Some(checkpointer)
            }
            None => None,
        };

        Learner::new(
            model,
            optim,
            self.num_epochs,
            callback,
            self.checkpoint,
            create_checkpointer(self.checkpointer_model),
            create_checkpointer(self.checkpointer_optimizer),
        )
    }
}
