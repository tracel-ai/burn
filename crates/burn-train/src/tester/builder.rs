use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use super::Learner;
use crate::components_test::LearnerComponentsMarker;
use crate::logger_test::{FileMetricLogger, MetricLogger};
use crate::metric_test::processor::{FullEventProcessor, Metrics};
use crate::metric_test::store::{EventStoreClient, LogEventStore};
use crate::metric_test::{Adaptor, Metric};
use crate::renderer_test::{default_renderer, MetricsRenderer};
use crate::tester::base::TrainingInterrupter;
use crate::{ApplicationLoggerInstaller, FileApplicationLoggerInstaller, LearnerSummaryConfig};
use burn_core::module::AutodiffModule;
use burn_core::tensor::backend::AutodiffBackend;

/// Struct to configure and create a [learner](Learner).
pub struct LearnerBuilder<B, T>
where
    T: Send + 'static,
    B: AutodiffBackend,
{
    directory: PathBuf,
    devices: Vec<B::Device>,
    renderer: Option<Box<dyn MetricsRenderer + 'static>>,
    metrics: Metrics<T>,
    event_store: LogEventStore,
    interrupter: TrainingInterrupter,
    tracing_logger: Option<Box<dyn ApplicationLoggerInstaller>>,
    num_loggers: usize,
    summary_metrics: HashSet<String>,
    summary: bool,
}

impl<B, T> LearnerBuilder<B, T>
where
    B: AutodiffBackend,
    T: Send + 'static,
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
            summary_metrics: HashSet::new(),
            summary: false,
        }
    }

    /// Replace the default metric loggers with the provided ones.
    ///
    /// # Arguments
    ///
    /// * `logger_train` - The training logger.
    pub fn metric_loggers<MT, MV>(mut self, logger_train: MT) -> Self
    where
        MT: MetricLogger + 'static,
        MV: MetricLogger + 'static,
    {
        self.event_store.register_logger_train(logger_train);
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

    /// Register a [numeric](crate::metric::Numeric) training [metric](Metric).
    pub fn metric_train_numeric<Me>(mut self, metric: Me) -> Self
    where
        Me: Metric + crate::metric_test::Numeric + 'static,
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
    ) -> Learner<LearnerComponentsMarker<B, M, FullEventProcessor<T>>>
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
            .unwrap_or_else(|| default_renderer(self.interrupter.clone()));

        if self.num_loggers == 0 {
            self.event_store
                .register_logger_train(FileMetricLogger::new(self.directory.join("train")));
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

        Learner {
            model,
            event_processor,
            event_store,
            devices: self.devices,
            interrupter: self.interrupter,
            summary,
        }
    }
}
