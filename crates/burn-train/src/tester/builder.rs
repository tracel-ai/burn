use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use super::Tester;
use crate::components_test::TesterComponentsMarker;
use crate::learner::base::TrainingInterrupter;
use crate::learner::EarlyStoppingStrategy;
use crate::logger::{FileMetricLogger, MetricLogger};
use crate::metric::processor::{FullEventProcessor, Metrics};
use crate::metric::store::{EventStoreClient, LogEventStore};
use crate::metric::{Adaptor, Metric};
use crate::renderer::{default_renderer, MetricsRenderer};
use crate::{
    ApplicationLoggerInstaller, FileApplicationLoggerInstaller, Learner, LearnerSummaryConfig,
};
use burn_core::module::AutodiffModule;
use burn_core::optim::AdamConfig;
use burn_core::tensor::backend::AutodiffBackend;

/// Struct to configure and create a [learner](Learner).
pub struct TesterBuilder<B, T, V>
where
    T: Send + 'static,
    V: Send + 'static,
    B: AutodiffBackend,
{
    directory: PathBuf,
    devices: Vec<B::Device>,
    renderer: Option<Box<dyn MetricsRenderer + 'static>>,
    metrics: Metrics<T, V>,
    event_store: LogEventStore,
    interrupter: TrainingInterrupter,
    tracing_logger: Option<Box<dyn ApplicationLoggerInstaller>>,
    num_loggers: usize,
    early_stopping: Option<Box<dyn EarlyStoppingStrategy>>,
    summary_metrics: HashSet<String>,
    summary: bool,
}

impl<B, T, V> TesterBuilder<B, T, V>
where
    B: AutodiffBackend,
    T: Send + 'static,
    V: Send + 'static,
{
    /// Creates a new learner builder.
    ///
    /// # Arguments
    ///
    /// * `directory` - The directory to save the checkpoints.
    pub fn new(directory: impl AsRef<Path>) -> Self {
        let directory = directory.as_ref().to_path_buf();
        let experiment_log_file = directory.join("experiment.log");
        Self {
            directory,
            devices: vec![B::Device::default()],
            metrics: Metrics::default(),
            event_store: LogEventStore::default(),
            renderer: None,
            interrupter: TrainingInterrupter::new(),
            tracing_logger: Some(Box::new(FileApplicationLoggerInstaller::new(
                experiment_log_file,
            ))),
            num_loggers: 0,
            early_stopping: None,
            summary_metrics: HashSet::new(),
            summary: false,
        }
    }

    /// Replace the default metric loggers with the provided ones.
    ///
    /// # Arguments
    ///
    /// * `logger_train` - The training logger.
    /// * `logger_valid` - The validation logger.
    pub fn metric_loggers<MT, MV>(mut self, logger_train: MT, logger_valid: MV) -> Self
    where
        MT: MetricLogger + 'static,
        MV: MetricLogger + 'static,
    {
        self.event_store.register_logger_train(logger_train);
        self.event_store.register_logger_valid(logger_valid);
        self.num_loggers += 1;
        self
    }

    /// Replace the default CLI renderer with a custom one.
    ///
    /// # Arguments
    ///
    /// * `renderer` - The custom renderer.
    pub fn renderer<MR>(mut self, renderer: MR) -> Self
    where
        MR: MetricsRenderer + 'static,
    {
        self.renderer = Some(Box::new(renderer));
        self
    }

    /// Register a training metric.
    pub fn metric_train<Me: Metric + 'static>(mut self, metric: Me) -> Self
    where
        T: Adaptor<Me::Input>,
    {
        self.metrics.register_train_metric(metric);
        self
    }

    /// Register a [numeric](crate::metric_test::Numeric) training [metric](Metric).
    pub fn metric_train_numeric<Me>(mut self, metric: Me) -> Self
    where
        Me: Metric + crate::metric::Numeric + 'static,
        T: Adaptor<Me::Input>,
    {
        self.summary_metrics.insert(Me::NAME.to_string());
        self.metrics.register_train_metric_numeric(metric);
        self
    }

    /// Run the training loop on multiple devices.
    pub fn devices(mut self, devices: Vec<B::Device>) -> Self {
        self.devices = devices;
        self
    }

    /// Provides a handle that can be used to interrupt training.
    pub fn interrupter(&self) -> TrainingInterrupter {
        self.interrupter.clone()
    }

    /// Register an [early stopping strategy](EarlyStoppingStrategy) to stop the training when the
    /// conditions are meet.
    pub fn early_stopping<Strategy>(mut self, strategy: Strategy) -> Self
    where
        Strategy: EarlyStoppingStrategy + 'static,
    {
        self.early_stopping = Some(Box::new(strategy));
        self
    }

    /// By default, Rust logs are captured and written into
    /// `experiment.log`. If disabled, standard Rust log handling
    /// will apply.
    pub fn with_application_logger(
        mut self,
        logger: Option<Box<dyn ApplicationLoggerInstaller>>,
    ) -> Self {
        self.tracing_logger = logger;
        self
    }

    /// Enable the training summary report.
    ///
    /// The summary will be displayed at the end of `.fit()`.
    pub fn summary(mut self) -> Self {
        self.summary = true;
        self
    }

    /// Create the [learner](Learner) from a [model](AutodiffModule) and an [optimizer](Optimizer).
    /// The [learning rate scheduler](LrScheduler) can also be a simple
    /// [learning rate](burn_core::LearningRate).
    #[allow(clippy::type_complexity)] // The goal for the builder is to handle all types and
                                      // creates a clean learner.
    pub fn build<M>(
        mut self,
        model: M,
        // the trait `LearnerComponents` is not implemented for `TesterComponentsMarker<B, M, full::FullEventProcessor<T, V>>`
    ) -> Tester<TesterComponentsMarker<B, M, FullEventProcessor<T, V>>>
    where
        M::Record: 'static,
        M: AutodiffModule<B> + core::fmt::Display + 'static,
    {
        if self.tracing_logger.is_some() {
            if let Err(e) = self.tracing_logger.as_ref().unwrap().install() {
                log::warn!("Failed to install the experiment logger: {}", e);
            }
        }
        let renderer = self
            .renderer
            .unwrap_or_else(|| default_renderer(self.interrupter.clone(), None));

        if self.num_loggers == 0 {
            self.event_store
                .register_logger_train(FileMetricLogger::new(self.directory.join("test")));
        }

        let event_store = Rc::new(EventStoreClient::new(self.event_store));
        let event_processor = FullEventProcessor::new(self.metrics, renderer, event_store.clone());

        let summary = if self.summary {
            Some(LearnerSummaryConfig {
                directory: self.directory,
                metrics: self.summary_metrics.into_iter().collect::<Vec<_>>(),
            })
        } else {
            None
        };

        let learner = Learner {
            model,
            optim: AdamConfig::new().init(),
            lr_scheduler: 0.0,
            checkpointer: None,
            num_epochs: 1,
            event_processor,
            event_store,
            checkpoint: None,
            grad_accumulation: None,
            devices: self.devices,
            interrupter: self.interrupter,
            early_stopping: self.early_stopping,
            summary,
        };

        Tester { learner }
    }
}
