use std::collections::HashSet;
use std::rc::Rc;

use super::Learner;
use crate::checkpoint::{
    AsyncCheckpointer, CheckpointingStrategy, ComposedCheckpointingStrategy, FileCheckpointer,
    KeepLastNCheckpoints, MetricCheckpointingStrategy,
};
use crate::components::LearnerComponentsMarker;
use crate::learner::base::TrainingInterrupter;
use crate::learner::EarlyStoppingStrategy;
use crate::logger::{FileMetricLogger, MetricLogger};
use crate::metric::processor::{FullEventProcessor, Metrics};
use crate::metric::store::{Aggregate, Direction, EventStoreClient, LogEventStore, Split};
use crate::metric::{Adaptor, LossMetric, Metric};
use crate::renderer::{default_renderer, MetricsRenderer};
use crate::{
    ApplicationLoggerInstaller, FileApplicationLoggerInstaller, LearnerCheckpointer,
    LearnerSummaryConfig,
};
use burn_core::lr_scheduler::LrScheduler;
use burn_core::module::AutodiffModule;
use burn_core::optim::Optimizer;
use burn_core::record::FileRecorder;
use burn_core::tensor::backend::AutodiffBackend;

/// Struct to configure and create a [learner](Learner).
pub struct LearnerBuilder<B, T, V, M, O, S>
where
    T: Send + 'static,
    V: Send + 'static,
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    O: Optimizer<M, B>,
    S: LrScheduler<B>,
{
    // Not that complex and very convenient when the traits are
    // already constrained correctly. Extracting in another type
    // would be more complex.
    #[allow(clippy::type_complexity)]
    checkpointers: Option<(
        AsyncCheckpointer<M::Record, B>,
        AsyncCheckpointer<O::Record, B>,
        AsyncCheckpointer<S::Record, B>,
    )>,
    num_epochs: usize,
    checkpoint: Option<usize>,
    directory: String,
    grad_accumulation: Option<usize>,
    devices: Vec<B::Device>,
    renderer: Option<Box<dyn MetricsRenderer + 'static>>,
    metrics: Metrics<T, V>,
    event_store: LogEventStore,
    interrupter: TrainingInterrupter,
    tracing_logger: Option<Box<dyn ApplicationLoggerInstaller>>,
    num_loggers: usize,
    checkpointer_strategy: Box<dyn CheckpointingStrategy>,
    early_stopping: Option<Box<dyn EarlyStoppingStrategy>>,
    summary_metrics: HashSet<String>,
    summary: bool,
}

impl<B, T, V, M, O, S> LearnerBuilder<B, T, V, M, O, S>
where
    B: AutodiffBackend,
    T: Send + 'static,
    V: Send + 'static,
    M: AutodiffModule<B> + core::fmt::Display + 'static,
    O: Optimizer<M, B>,
    S: LrScheduler<B>,
{
    /// Creates a new learner builder.
    ///
    /// # Arguments
    ///
    /// * `directory` - The directory to save the checkpoints.
    pub fn new(directory: &str) -> Self {
        Self {
            num_epochs: 1,
            checkpoint: None,
            checkpointers: None,
            directory: directory.to_string(),
            grad_accumulation: None,
            devices: vec![B::Device::default()],
            metrics: Metrics::default(),
            event_store: LogEventStore::default(),
            renderer: None,
            interrupter: TrainingInterrupter::new(),
            tracing_logger: Some(Box::new(FileApplicationLoggerInstaller::new(
                format!("{}/experiment.log", directory).as_str(),
            ))),
            num_loggers: 0,
            checkpointer_strategy: Box::new(
                ComposedCheckpointingStrategy::builder()
                    .add(KeepLastNCheckpoints::new(2))
                    .add(MetricCheckpointingStrategy::new::<LossMetric<B>>(
                        Aggregate::Mean,
                        Direction::Lowest,
                        Split::Valid,
                    ))
                    .build(),
            ),
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

    /// Update the checkpointing_strategy.
    pub fn with_checkpointing_strategy<CS>(mut self, strategy: CS) -> Self
    where
        CS: CheckpointingStrategy + 'static,
    {
        self.checkpointer_strategy = Box::new(strategy);
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

    /// Register a validation metric.
    pub fn metric_valid<Me: Metric + 'static>(mut self, metric: Me) -> Self
    where
        V: Adaptor<Me::Input>,
    {
        self.metrics.register_valid_metric(metric);
        self
    }

    /// Enable gradients accumulation.
    ///
    /// # Notes
    ///
    /// When you enable gradients accumulation, the gradients object used by the optimizer will be
    /// the sum of all gradients generated by each backward pass. It might be a good idea to
    /// reduce the learning to compensate.
    ///
    /// The effect is similar to increasing the `batch size` and the `learning rate` by the `accumulation`
    /// amount.
    pub fn grads_accumulation(mut self, accumulation: usize) -> Self {
        self.grad_accumulation = Some(accumulation);
        self
    }

    /// Register a [numeric](crate::metric::Numeric) training [metric](Metric).
    pub fn metric_train_numeric<Me>(mut self, metric: Me) -> Self
    where
        Me: Metric + crate::metric::Numeric + 'static,
        T: Adaptor<Me::Input>,
    {
        self.summary_metrics.insert(Me::NAME.to_string());
        self.metrics.register_train_metric_numeric(metric);
        self
    }

    /// Register a [numeric](crate::metric::Numeric) validation [metric](Metric).
    pub fn metric_valid_numeric<Me: Metric + crate::metric::Numeric + 'static>(
        mut self,
        metric: Me,
    ) -> Self
    where
        V: Adaptor<Me::Input>,
    {
        self.summary_metrics.insert(Me::NAME.to_string());
        self.metrics.register_valid_metric_numeric(metric);
        self
    }

    /// The number of epochs the training should last.
    pub fn num_epochs(mut self, num_epochs: usize) -> Self {
        self.num_epochs = num_epochs;
        self
    }

    /// Run the training loop on multiple devices.
    pub fn devices(mut self, devices: Vec<B::Device>) -> Self {
        self.devices = devices;
        self
    }

    /// The epoch from which the training must resume.
    pub fn checkpoint(mut self, checkpoint: usize) -> Self {
        self.checkpoint = Some(checkpoint);
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

    /// Register a checkpointer that will save the [optimizer](Optimizer), the
    /// [model](AutodiffModule) and the [scheduler](LrScheduler) to different files.
    pub fn with_file_checkpointer<FR>(mut self, recorder: FR) -> Self
    where
        FR: FileRecorder<B> + 'static,
        FR: FileRecorder<B::InnerBackend> + 'static,
        O::Record: 'static,
        M::Record: 'static,
        S::Record: 'static,
    {
        let checkpointer_model = FileCheckpointer::new(
            recorder.clone(),
            format!("{}/checkpoint", self.directory).as_str(),
            "model",
        );
        let checkpointer_optimizer = FileCheckpointer::new(
            recorder.clone(),
            format!("{}/checkpoint", self.directory).as_str(),
            "optim",
        );
        let checkpointer_scheduler: FileCheckpointer<FR> = FileCheckpointer::new(
            recorder,
            format!("{}/checkpoint", self.directory).as_str(),
            "scheduler",
        );

        self.checkpointers = Some((
            AsyncCheckpointer::new(checkpointer_model),
            AsyncCheckpointer::new(checkpointer_optimizer),
            AsyncCheckpointer::new(checkpointer_scheduler),
        ));

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
    pub fn build(
        mut self,
        model: M,
        optim: O,
        lr_scheduler: S,
    ) -> Learner<
        LearnerComponentsMarker<
            B,
            S,
            M,
            O,
            AsyncCheckpointer<M::Record, B>,
            AsyncCheckpointer<O::Record, B>,
            AsyncCheckpointer<S::Record, B>,
            FullEventProcessor<T, V>,
            Box<dyn CheckpointingStrategy>,
        >,
    >
    where
        M::Record: 'static,
        O::Record: 'static,
        S::Record: 'static,
    {
        if self.tracing_logger.is_some() {
            if let Err(e) = self.tracing_logger.as_ref().unwrap().install() {
                log::warn!("Failed to install the experiment logger: {}", e);
            }
        }
        let renderer = self.renderer.unwrap_or_else(|| {
            Box::new(default_renderer(self.interrupter.clone(), self.checkpoint))
        });
        let directory = &self.directory;

        if self.num_loggers == 0 {
            self.event_store
                .register_logger_train(FileMetricLogger::new(
                    format!("{directory}/train").as_str(),
                ));
            self.event_store
                .register_logger_valid(FileMetricLogger::new(
                    format!("{directory}/valid").as_str(),
                ));
        }

        let event_store = Rc::new(EventStoreClient::new(self.event_store));
        let event_processor = FullEventProcessor::new(self.metrics, renderer, event_store.clone());

        let checkpointer = self.checkpointers.map(|(model, optim, scheduler)| {
            LearnerCheckpointer::new(model, optim, scheduler, self.checkpointer_strategy)
        });

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
            optim,
            lr_scheduler,
            checkpointer,
            num_epochs: self.num_epochs,
            event_processor,
            event_store,
            checkpoint: self.checkpoint,
            grad_accumulation: self.grad_accumulation,
            devices: self.devices,
            interrupter: self.interrupter,
            early_stopping: self.early_stopping,
            summary,
        }
    }
}
