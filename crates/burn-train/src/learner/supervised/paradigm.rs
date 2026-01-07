use crate::checkpoint::{
    AsyncCheckpointer, CheckpointingStrategy, ComposedCheckpointingStrategy, FileCheckpointer,
    KeepLastNCheckpoints, MetricCheckpointingStrategy,
};
use crate::components::{InferenceModelOutput, TrainingModelOutput};
use crate::learner::EarlyStoppingStrategy;
use crate::learner::base::Interrupter;
use crate::logger::{FileMetricLogger, MetricLogger};
use crate::metric::processor::{
    AsyncProcessorTraining, FullEventProcessorTraining, ItemLazy, MetricsTraining,
};
use crate::metric::store::{Aggregate, Direction, EventStoreClient, LogEventStore, Split};
use crate::metric::{Adaptor, LossMetric, Metric, Numeric};
use crate::multi::MultiDeviceLearningStrategy;
use crate::renderer::{MetricsRenderer, default_renderer};
use crate::single::SingleDevicetrainingStrategy;
use crate::{
    ApplicationLoggerInstaller, EarlyStoppingStrategyRef, FileApplicationLoggerInstaller,
    InferenceBackend, InferenceModel, InferenceModelInput, LearnerModelRecord,
    LearnerOptimizerRecord, LearnerSchedulerRecord, LearnerSummaryConfig, LearningCheckpointer,
    LearningComponentsMarker, LearningComponentsTypes, LearningResult, ModelDataMarker, TrainStep,
    TrainingBackend, TrainingComponents, TrainingModelInput, TrainingStrategy, ValidStep,
};
use crate::{Learner, SupervisedLearningStrategy};
use burn_core::data::dataloader::DataLoader;
use burn_core::module::AutodiffModule;
use burn_core::record::FileRecorder;
use burn_core::tensor::backend::AutodiffBackend;
use burn_optim::Optimizer;
use burn_optim::lr_scheduler::LrScheduler;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// A reference to the training split [DataLoader](DataLoader).
pub type TrainLoader<LC> = Arc<dyn DataLoader<TrainingBackend<LC>, TrainingModelInput<LC>>>;
/// A reference to the validation split [DataLoader](DataLoader).
pub type ValidLoader<LC> = Arc<dyn DataLoader<InferenceBackend<LC>, InferenceModelInput<LC>>>;
/// The event processor type for supervised learning.
pub type SupervisedTrainingEventProcessor<LC> = AsyncProcessorTraining<
    FullEventProcessorTraining<TrainingModelOutput<LC>, InferenceModelOutput<LC>>,
>;

/// Structure to configure and launch supervised learning trainings.
pub struct SupervisedTraining<LC>
where
    LC: LearningComponentsTypes,
{
    checkpointers: Option<(
        AsyncCheckpointer<LearnerModelRecord<LC>, TrainingBackend<LC>>,
        AsyncCheckpointer<LearnerOptimizerRecord<LC>, TrainingBackend<LC>>,
        AsyncCheckpointer<LearnerSchedulerRecord<LC>, TrainingBackend<LC>>,
    )>,
    num_epochs: usize,
    checkpoint: Option<usize>,
    directory: PathBuf,
    grad_accumulation: Option<usize>,
    renderer: Option<Box<dyn MetricsRenderer + 'static>>,
    metrics: MetricsTraining<TrainingModelOutput<LC>, InferenceModelOutput<LC>>,
    event_store: LogEventStore,
    interrupter: Interrupter,
    tracing_logger: Option<Box<dyn ApplicationLoggerInstaller>>,
    checkpointer_strategy: Box<dyn CheckpointingStrategy>,
    early_stopping: Option<EarlyStoppingStrategyRef>,
    training_strategy: TrainingStrategy<LC>,
    dataloader_train: TrainLoader<LC>,
    dataloader_valid: ValidLoader<LC>,
    // Use BTreeSet instead of HashSet for consistent (alphabetical) iteration order
    summary_metrics: BTreeSet<String>,
    summary: bool,
}

impl<B, LR, M, O, TI, TO, VI, VO>
    SupervisedTraining<LearningComponentsMarker<B, LR, M, O, ModelDataMarker<TI, VI, TO, VO>>>
where
    B: AutodiffBackend,
    LR: LrScheduler + 'static,
    M: TrainStep<TI, TO> + AutodiffModule<B> + core::fmt::Display + 'static,
    M::InnerModule: ValidStep<VI, VO>,
    O: Optimizer<M, B> + 'static,
    TI: Send + 'static,
    VI: Send + 'static,
    TO: ItemLazy + 'static,
    VO: ItemLazy + 'static,
{
    /// Creates a new runner for a supervised training.
    ///
    /// # Arguments
    ///
    /// * `directory` - The directory to save the checkpoints.
    /// * `dataloader_train` - The dataloader for the training split.
    /// * `dataloader_valid` - The dataloader for the validation split.
    pub fn new(
        directory: impl AsRef<Path>,
        dataloader_train: Arc<dyn DataLoader<B, TI>>,
        dataloader_valid: Arc<dyn DataLoader<B::InnerBackend, VI>>,
    ) -> Self {
        let directory = directory.as_ref().to_path_buf();
        let experiment_log_file = directory.join("experiment.log");
        Self {
            num_epochs: 1,
            checkpoint: None,
            checkpointers: None,
            directory,
            grad_accumulation: None,
            metrics: MetricsTraining::default(),
            event_store: LogEventStore::default(),
            renderer: None,
            interrupter: Interrupter::new(),
            tracing_logger: Some(Box::new(FileApplicationLoggerInstaller::new(
                experiment_log_file,
            ))),
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
            training_strategy: TrainingStrategy::SingleDevice(Default::default()),
            summary_metrics: BTreeSet::new(),
            summary: false,
            dataloader_train,
            dataloader_valid,
        }
    }
}

impl<LC: LearningComponentsTypes> SupervisedTraining<LC> {
    /// Replace the default training strategy (SingleDeviceTrainingStrategy) with the provided ones.
    ///
    /// # Arguments
    ///
    /// * `training_strategy` - The training strategy.
    pub fn with_training_strategy(mut self, training_strategy: TrainingStrategy<LC>) -> Self {
        self.training_strategy = training_strategy;
        self
    }

    /// Replace the default metric loggers with the provided ones.
    ///
    /// # Arguments
    ///
    /// * `logger` - The training logger.
    pub fn with_metric_logger<ML>(mut self, logger: ML) -> Self
    where
        ML: MetricLogger + 'static,
    {
        self.event_store.register_logger(logger);
        self
    }

    /// Update the checkpointing_strategy.
    pub fn with_checkpointing_strategy<CS: CheckpointingStrategy + 'static>(
        mut self,
        strategy: CS,
    ) -> Self {
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

    /// Register all metrics as numeric for the training and validation set.
    pub fn metrics<Me: MetricRegistration<LC>>(self, metrics: Me) -> Self {
        metrics.register(self)
    }

    /// Register all metrics as numeric for the training and validation set.
    pub fn metrics_text<Me: TextMetricRegistration<LC>>(self, metrics: Me) -> Self {
        metrics.register(self)
    }

    /// Register a training metric.
    pub fn metric_train<Me: Metric + 'static>(mut self, metric: Me) -> Self
    where
        <TrainingModelOutput<LC> as ItemLazy>::ItemSync: Adaptor<Me::Input>,
    {
        self.metrics.register_train_metric(metric);
        self
    }

    /// Register a validation metric.
    pub fn metric_valid<Me: Metric + 'static>(mut self, metric: Me) -> Self
    where
        <InferenceModelOutput<LC> as ItemLazy>::ItemSync: Adaptor<Me::Input>,
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
        Me: Metric + Numeric + 'static,
        <TrainingModelOutput<LC> as ItemLazy>::ItemSync: Adaptor<Me::Input>,
    {
        self.summary_metrics.insert(metric.name().to_string());
        self.metrics.register_train_metric_numeric(metric);
        self
    }

    /// Register a [numeric](crate::metric::Numeric) validation [metric](Metric).
    pub fn metric_valid_numeric<Me: Metric + Numeric + 'static>(mut self, metric: Me) -> Self
    where
        <InferenceModelOutput<LC> as ItemLazy>::ItemSync: Adaptor<Me::Input>,
    {
        self.summary_metrics.insert(metric.name().to_string());
        self.metrics.register_valid_metric_numeric(metric);
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

    /// Provides a handle that can be used to interrupt training.
    pub fn interrupter(&self) -> Interrupter {
        self.interrupter.clone()
    }

    /// Override the handle for stopping training with an externally provided handle
    pub fn with_interrupter(mut self, interrupter: Interrupter) -> Self {
        self.interrupter = interrupter;
        self
    }

    /// Register an [early stopping strategy](EarlyStoppingStrategy) to stop the training when the
    /// conditions are meet.
    pub fn early_stopping<Strategy>(mut self, strategy: Strategy) -> Self
    where
        Strategy: EarlyStoppingStrategy + Clone + Send + Sync + 'static,
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
        FR: FileRecorder<<LC as LearningComponentsTypes>::Backend> + 'static,
        FR: FileRecorder<
                <<LC as LearningComponentsTypes>::Backend as AutodiffBackend>::InnerBackend,
            > + 'static,
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
    /// The summary will be displayed after `.fit()`, when the renderer is dropped.
    pub fn summary(mut self) -> Self {
        self.summary = true;
        self
    }
}

impl<LC: LearningComponentsTypes + Send + 'static> SupervisedTraining<LC> {
    /// Launch this training with the given [Learner](Learner).
    pub fn launch(mut self, learner: Learner<LC>) -> LearningResult<InferenceModel<LC>> {
        if self.tracing_logger.is_some()
            && let Err(e) = self.tracing_logger.as_ref().unwrap().install()
        {
            log::warn!("Failed to install the experiment logger: {e}");
        }
        let renderer = self
            .renderer
            .unwrap_or_else(|| default_renderer(self.interrupter.clone(), self.checkpoint));

        if !self.event_store.has_loggers() {
            self.event_store
                .register_logger(FileMetricLogger::new(self.directory.clone()));
        }

        let event_store = Arc::new(EventStoreClient::new(self.event_store));
        let event_processor = AsyncProcessorTraining::new(FullEventProcessorTraining::new(
            self.metrics,
            renderer,
            event_store.clone(),
        ));

        let checkpointer = self.checkpointers.map(|(model, optim, scheduler)| {
            LearningCheckpointer::new(
                model.with_interrupter(self.interrupter.clone()),
                optim.with_interrupter(self.interrupter.clone()),
                scheduler.with_interrupter(self.interrupter.clone()),
                self.checkpointer_strategy,
            )
        });

        let summary = if self.summary {
            Some(LearnerSummaryConfig {
                directory: self.directory,
                metrics: self.summary_metrics.into_iter().collect::<Vec<_>>(),
            })
        } else {
            None
        };

        let components = TrainingComponents {
            checkpoint: self.checkpoint,
            checkpointer,
            interrupter: self.interrupter,
            early_stopping: self.early_stopping,
            event_processor,
            event_store,
            num_epochs: self.num_epochs,
            grad_accumulation: self.grad_accumulation,
            summary,
        };

        match self.training_strategy {
            TrainingStrategy::SingleDevice(device) => {
                let single_device: SingleDevicetrainingStrategy<LC> =
                    SingleDevicetrainingStrategy::new(device);
                single_device.train(
                    learner,
                    self.dataloader_train,
                    self.dataloader_valid,
                    components,
                )
            }
            TrainingStrategy::Custom(learning_paradigm) => learning_paradigm.train(
                learner,
                self.dataloader_train,
                self.dataloader_valid,
                components,
            ),
            TrainingStrategy::MultiDevice(devices, multi_device_optim) => {
                let multi_device = MultiDeviceLearningStrategy::new(devices, multi_device_optim);
                multi_device.train(
                    learner,
                    self.dataloader_train,
                    self.dataloader_valid,
                    components,
                )
            }
            #[cfg(feature = "ddp")]
            TrainingStrategy::DistributedDataParallel { devices, config } => {
                use crate::ddp::DdpTrainingStrategy;

                let ddp = DdpTrainingStrategy::new(devices.clone(), config.clone());
                ddp.train(
                    learner,
                    self.dataloader_train,
                    self.dataloader_valid,
                    components,
                )
            }
        }
    }
}

/// Trait to fake variadic generics.
pub trait MetricRegistration<LC: LearningComponentsTypes>: Sized {
    /// Register the metrics.
    fn register(self, builder: SupervisedTraining<LC>) -> SupervisedTraining<LC>;
}

/// Trait to fake variadic generics.
pub trait TextMetricRegistration<LC: LearningComponentsTypes>: Sized {
    /// Register the metrics.
    fn register(self, builder: SupervisedTraining<LC>) -> SupervisedTraining<LC>;
}

macro_rules! gen_tuple {
    ($($M:ident),*) => {
        impl<$($M,)* LC: LearningComponentsTypes> TextMetricRegistration<LC> for ($($M,)*)
        where
            $(<TrainingModelOutput<LC> as ItemLazy>::ItemSync: Adaptor<$M::Input>,)*
            $(<InferenceModelOutput<LC> as ItemLazy>::ItemSync: Adaptor<$M::Input>,)*
            $($M: Metric + 'static,)*
        {
            #[allow(non_snake_case)]
            fn register(
                self,
                builder: SupervisedTraining<LC>,
            ) -> SupervisedTraining<LC> {
                let ($($M,)*) = self;
                $(let builder = builder.metric_train($M.clone());)*
                $(let builder = builder.metric_valid($M);)*
                builder
            }
        }

        impl<$($M,)* LC: LearningComponentsTypes> MetricRegistration<LC> for ($($M,)*)
        where
            $(<TrainingModelOutput<LC> as ItemLazy>::ItemSync: Adaptor<$M::Input>,)*
            $(<InferenceModelOutput<LC> as ItemLazy>::ItemSync: Adaptor<$M::Input>,)*
            $($M: Metric + Numeric + 'static,)*
        {
            #[allow(non_snake_case)]
            fn register(
                self,
                builder: SupervisedTraining<LC>,
            ) -> SupervisedTraining<LC> {
                let ($($M,)*) = self;
                $(let builder = builder.metric_train_numeric($M.clone());)*
                $(let builder = builder.metric_valid_numeric($M);)*
                builder
            }
        }
    };
}

gen_tuple!(M1);
gen_tuple!(M1, M2);
gen_tuple!(M1, M2, M3);
gen_tuple!(M1, M2, M3, M4);
gen_tuple!(M1, M2, M3, M4, M5);
gen_tuple!(M1, M2, M3, M4, M5, M6);
