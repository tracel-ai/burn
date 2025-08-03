use std::collections::BTreeSet;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::Learner;
use crate::checkpoint::{
    AsyncCheckpointer, CheckpointingStrategy, ComposedCheckpointingStrategy, FileCheckpointer,
    KeepLastNCheckpoints, MetricCheckpointingStrategy,
};
use crate::components::{LearnerComponentsMarker, LearningDataMarker};
use crate::learner::EarlyStoppingStrategy;
use crate::learner::base::TrainingInterrupter;
use crate::logger::{FileMetricLogger, MetricLogger};
use crate::metric::processor::{AsyncProcessor, FullEventProcessor, ItemLazy, Metrics};
use crate::metric::store::{Aggregate, Direction, EventStoreClient, LogEventStore, Split};
use crate::metric::{Adaptor, LossMetric, Metric};
use crate::renderer::{MetricsRenderer, default_renderer};
use crate::{
    ApplicationLoggerInstaller, FileApplicationLoggerInstaller, LearnerCheckpointer,
    LearnerSummaryConfig, LearningStrategy, TrainStep, ValidStep,
};
use burn_core::lr_scheduler::LrScheduler;
use burn_core::module::AutodiffModule;
use burn_core::optim::Optimizer;
use burn_core::record::FileRecorder;
use burn_core::tensor::backend::AutodiffBackend;

/// Struct to configure and create a [learner](Learner).
///
/// The generics components of the builder should probably not be set manually, as they are
/// optimized for Rust type inference.
pub struct LearnerBuilder<B, M, O, S, TI, VI, TO, VO>
where
    B: AutodiffBackend,
    M: AutodiffModule<B> + TrainStep<TI, TO> + core::fmt::Display + 'static,
    M::InnerModule: ValidStep<VI, VO>,
    O: Optimizer<M, B>,
    S: LrScheduler,
    TI: Send + 'static,
    VI: Send + 'static,
    TO: ItemLazy + 'static,
    VO: ItemLazy + 'static,
{
    // Not that complex and very convenient when the traits are
    // already constrained correctly. Extracting in another type
    // would be more complex.
    #[allow(clippy::type_complexity)]
    checkpointers: Option<(
        AsyncCheckpointer<M::Record, B>,
        AsyncCheckpointer<O::Record, B>,
        AsyncCheckpointer<S::Record<B>, B>,
    )>,
    num_epochs: usize,
    checkpoint: Option<usize>,
    directory: PathBuf,
    grad_accumulation: Option<usize>,
    learning_strategy: LearningStrategy<B>,
    renderer: Option<Box<dyn MetricsRenderer + 'static>>,
    metrics: Metrics<TO, VO>,
    event_store: LogEventStore,
    interrupter: TrainingInterrupter,
    tracing_logger: Option<Box<dyn ApplicationLoggerInstaller>>,
    num_loggers: usize,
    checkpointer_strategy: Box<dyn CheckpointingStrategy>,
    early_stopping: Option<Box<dyn EarlyStoppingStrategy>>,
    // Use BTreeSet instead of HashSet for consistent (alphabetical) iteration order
    summary_metrics: BTreeSet<String>,
    summary: bool,
    _p: PhantomData<(TI, VI, TO, VO)>,
}

impl<B, M, O, S, TI, VI, TO, VO> LearnerBuilder<B, M, O, S, TI, VI, TO, VO>
where
    B: AutodiffBackend,
    M: AutodiffModule<B> + TrainStep<TI, TO> + core::fmt::Display + 'static,
    M::InnerModule: ValidStep<VI, VO>,
    O: Optimizer<M, B>,
    S: LrScheduler,
    TI: Send + 'static,
    VI: Send + 'static,
    TO: ItemLazy + 'static,
    VO: ItemLazy + 'static,
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
            num_epochs: 1,
            checkpoint: None,
            checkpointers: None,
            directory,
            grad_accumulation: None,
            learning_strategy: LearningStrategy::default(),
            metrics: Metrics::default(),
            event_store: LogEventStore::default(),
            renderer: None,
            interrupter: TrainingInterrupter::new(),
            tracing_logger: Some(Box::new(FileApplicationLoggerInstaller::new(
                experiment_log_file,
            ))),
            num_loggers: 0,
            checkpointer_strategy: Box::new(
                ComposedCheckpointingStrategy::builder()
                    .add(KeepLastNCheckpoints::new(2))
                    .add(MetricCheckpointingStrategy::new(
                        &LossMetric::<B>::new(), // default to valid loss
                        Aggregate::Mean,
                        Direction::Lowest,
                        Split::Valid,
                    ))
                    .build(),
            ),
            early_stopping: None,
            summary_metrics: BTreeSet::new(),
            summary: false,
            _p: PhantomData,
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
        <TO as ItemLazy>::ItemSync: Adaptor<Me::Input>,
    {
        self.metrics.register_train_metric(metric);
        self
    }

    /// Register a validation metric.
    pub fn metric_valid<Me: Metric + 'static>(mut self, metric: Me) -> Self
    where
        <VO as ItemLazy>::ItemSync: Adaptor<Me::Input>,
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
        <TO as ItemLazy>::ItemSync: Adaptor<Me::Input>,
    {
        self.summary_metrics.insert(metric.name());
        self.metrics.register_train_metric_numeric(metric);
        self
    }

    /// Register a [numeric](crate::metric::Numeric) validation [metric](Metric).
    pub fn metric_valid_numeric<Me: Metric + crate::metric::Numeric + 'static>(
        mut self,
        metric: Me,
    ) -> Self
    where
        <VO as ItemLazy>::ItemSync: Adaptor<Me::Input>,
    {
        self.summary_metrics.insert(metric.name());
        self.metrics.register_valid_metric_numeric(metric);
        self
    }

    /// The number of epochs the training should last.
    pub fn num_epochs(mut self, num_epochs: usize) -> Self {
        self.num_epochs = num_epochs;
        self
    }

    /// Run the training loop with different strategies
    pub fn learning_strategy(mut self, learning_strategy: LearningStrategy<B>) -> Self {
        self.learning_strategy = learning_strategy;
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
        S::Record<B>: 'static,
    {
        let checkpoint_dir = self.directory.join("checkpoint");
        let checkpointer_model = FileCheckpointer::new(recorder.clone(), &checkpoint_dir, "model");
        let checkpointer_optimizer =
            FileCheckpointer::new(recorder.clone(), &checkpoint_dir, "optim");
        let checkpointer_scheduler: FileCheckpointer<FR> =
            FileCheckpointer::new(recorder, &checkpoint_dir, "scheduler");

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
            AsyncCheckpointer<S::Record<B>, B>,
            AsyncProcessor<FullEventProcessor<TO, VO>>,
            Box<dyn CheckpointingStrategy>,
            LearningDataMarker<TI, VI, TO, VO>,
        >,
    >
    where
        M::Record: 'static,
        O::Record: 'static,
        S::Record<B>: 'static,
    {
        if self.tracing_logger.is_some() {
            if let Err(e) = self.tracing_logger.as_ref().unwrap().install() {
                log::warn!("Failed to install the experiment logger: {e}");
            }
        }
        let renderer = self
            .renderer
            .unwrap_or_else(|| default_renderer(self.interrupter.clone(), self.checkpoint));

        if self.num_loggers == 0 {
            self.event_store
                .register_logger_train(FileMetricLogger::new(self.directory.join("train")));
            self.event_store
                .register_logger_valid(FileMetricLogger::new(self.directory.join("valid")));
        }

        let event_store = Arc::new(EventStoreClient::new(self.event_store));
        let event_processor = AsyncProcessor::new(FullEventProcessor::new(
            self.metrics,
            renderer,
            event_store.clone(),
        ));

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

        let learning_strategy = Self::prepare_learning_strategy(self.learning_strategy);

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
            learning_strategy,
            interrupter: self.interrupter,
            early_stopping: self.early_stopping,
            summary,
        }
    }

    fn prepare_learning_strategy(learning_strategy: LearningStrategy<B>) -> LearningStrategy<B> {
        if let LearningStrategy::MultiDeviceNaive(devices) = &learning_strategy {
            if devices.len() == 1 {
                return LearningStrategy::SingleDevice(devices[0].clone());
            }
        }

        learning_strategy
    }
}
