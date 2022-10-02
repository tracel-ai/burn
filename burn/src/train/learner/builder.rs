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

/// Struct to configure and create a [learner](Learner).
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
    fn new(renderer: Box<dyn DashboardRenderer>) -> Self {
        Self {
            dashboard: Dashboard::new(renderer),
            num_epochs: 1,
            checkpoint: None,
            checkpointer_model: None,
            checkpointer_optimizer: None,
        }
    }

    /// Register a training metric.
    pub fn metric_train<M: Metric<T> + 'static>(mut self, metric: M) -> Self {
        self.dashboard.register_train(metric);
        self
    }

    /// Register a validation metric.
    pub fn metric_valid<M: Metric<V> + 'static>(mut self, metric: M) -> Self {
        self.dashboard.register_valid(metric);
        self
    }

    /// Register a training metric and displays it on a plot.
    ///
    /// # Notes
    ///
    /// Only [numeric](Numeric) metric can be displayed on a plot.
    /// If the same metric is also registered for the [validation split](Self::metric_valid_plot),
    /// the same graph will be used for both.
    pub fn metric_train_plot<M: Metric<T> + Numeric + 'static>(mut self, metric: M) -> Self {
        self.dashboard.register_train_plot(metric);
        self
    }

    /// Register a validation metric and displays it on a plot.
    ///
    /// # Notes
    ///
    /// Only [numeric](Numeric) metric can be displayed on a plot.
    /// If the same metric is also registered for the [training split](Self::metric_train_plot),
    /// the same graph will be used for both.
    pub fn metric_valid_plot<M: Metric<V> + Numeric + 'static>(mut self, metric: M) -> Self {
        self.dashboard.register_valid_plot(metric);
        self
    }

    /// The number of epochs the training should last.
    pub fn num_epochs(mut self, num_epochs: usize) -> Self {
        self.num_epochs = num_epochs;
        self
    }

    /// The epoch from which the training must resume.
    pub fn checkpoint(mut self, checkpoint: usize) -> Self {
        self.checkpoint = Some(checkpoint);
        self
    }

    /// Register a checkpointer that will save the [optimizer](crate::optim::Optimizer) and the
    /// [model](crate::module::Module) [states](crate::module::State) in the specified directoty.
    pub fn with_file_checkpointer<P: Element + serde::de::DeserializeOwned + serde::Serialize>(
        mut self,
        directory: &str,
    ) -> Self {
        self.checkpointer_model = Some(Arc::new(FileCheckpointer::<P>::new(directory, "model")));
        self.checkpointer_optimizer =
            Some(Arc::new(FileCheckpointer::<P>::new(directory, "optim")));
        self
    }

    /// Create the [learner](Learner) from a [module](ADModule) and an
    /// [optimizer](crate::optim::Optimizer).
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

        Learner {
            model,
            optim,
            num_epochs: self.num_epochs,
            callback,
            checkpoint: self.checkpoint,
            checkpointer_model: create_checkpointer(self.checkpointer_model),
            checkpointer_optimizer: create_checkpointer(self.checkpointer_optimizer),
        }
    }
}
