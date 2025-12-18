use crate::checkpoint::{
    AsyncCheckpointer, CheckpointingStrategy, ComposedCheckpointingStrategy, FileCheckpointer,
    KeepLastNCheckpoints, MetricCheckpointingStrategy,
};
use crate::components::{OutputTrain, OutputValid, TrainLoader, ValidLoader};
use crate::learner::EarlyStoppingStrategy;
use crate::learner::base::Interrupter;
use crate::logger::{FileMetricLogger, MetricLogger};
use crate::metric::processor::{
    AsyncProcessorTraining, FullEventProcessorTraining, ItemLazy, MetricsTraining,
};
use crate::metric::store::{Aggregate, Direction, EventStoreClient, LogEventStore, Split};
use crate::metric::{Adaptor, LossMetric, Metric, Numeric};
use crate::multi::MultiDeviceLearningStrategyV2;
use crate::renderer::{MetricsRenderer, default_renderer};
use crate::single::SingleDevicetrainingStrategy;
use crate::{
    ApplicationLoggerInstaller, EarlyStoppingStrategyRef, FileApplicationLoggerInstaller,
    LearnerSummaryConfig, LearningCheckpointer, LearningComponentsTypes, LearningDataMarker,
    LearningParadigm, ParadigmComponentsMarker, ParadigmComponentsTypes,
    SupervisedLearningComponentsMarker, SupervisedLearningComponentsTypes, TrainBackend, TrainStep,
    TrainingResult, TrainingStrategy, ValidStep,
};
use crate::{Learner, SupervisedLearningStrategy};
use burn_core::module::Module;
use burn_core::record::FileRecorder;
use burn_core::tensor::backend::AutodiffBackend;
use burn_optim::Optimizer;
use burn_optim::lr_scheduler::LrScheduler;
use burn_optim::lr_scheduler::composed::ComposedLrScheduler;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

impl<LC, TI, TO, VI, VO>
    SupervisedTraining<
        SupervisedLearningComponentsMarker<
            ParadigmComponentsMarker<
                LearningDataMarker<TI, VI, TO, VO>,
                AsyncProcessorTraining<FullEventProcessorTraining<TO, VO>>,
                Box<dyn CheckpointingStrategy>,
            >,
            LC,
            LearningDataMarker<TI, VI, TO, VO>,
            LC::Model,
            LC::Optimizer,
            ComposedLrScheduler,
        >,
    >
where
    LC: LearningComponentsTypes,
    LC::Model: TrainStep<TI, TO>,
    LC::InnerModel: ValidStep<VI, VO>,
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
        dataloader_train: TrainLoader<LC, LearningDataMarker<TI, VI, TO, VO>>,
        dataloader_valid: ValidLoader<LC, LearningDataMarker<TI, VI, TO, VO>>,
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
                        &LossMetric::<LC::Backend>::new(), // default to valid loss
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
            // learner,
        }
    }
}

/// Structure to configure and launch supervised learning trainings.
pub struct SupervisedTraining<SC: SupervisedLearningComponentsTypes> {
    // Not that complex and very convenient when the traits are
    // already constrained correctly. Extracting in another type
    // would be more complex.
    #[allow(clippy::type_complexity)]
    checkpointers: Option<(
        AsyncCheckpointer<
            <<SC::LC as LearningComponentsTypes>::Model as Module<TrainBackend<SC::LC>>>::Record,
            TrainBackend<SC::LC>,
        >,
        AsyncCheckpointer<
            <<SC::LC as LearningComponentsTypes>::Optimizer as Optimizer<
                SC::Model,
                TrainBackend<SC::LC>,
            >>::Record,
            TrainBackend<SC::LC>,
        >,
        AsyncCheckpointer<
            <<SC::LC as LearningComponentsTypes>::LrScheduler as LrScheduler>::Record<
                TrainBackend<SC::LC>,
            >,
            TrainBackend<SC::LC>,
        >,
    )>,
    num_epochs: usize,
    checkpoint: Option<usize>,
    directory: PathBuf,
    grad_accumulation: Option<usize>,
    renderer: Option<Box<dyn MetricsRenderer + 'static>>,
    metrics: MetricsTraining<OutputTrain<SC::LD>, OutputValid<SC::LD>>,
    event_store: LogEventStore,
    interrupter: Interrupter,
    tracing_logger: Option<Box<dyn ApplicationLoggerInstaller>>,
    checkpointer_strategy: Box<dyn CheckpointingStrategy>,
    early_stopping: Option<EarlyStoppingStrategyRef>,
    training_strategy: TrainingStrategy<SC>,
    dataloader_train: TrainLoader<SC::LC, SC::LD>,
    dataloader_valid: ValidLoader<SC::LC, SC::LD>,
    // Use BTreeSet instead of HashSet for consistent (alphabetical) iteration order
    summary_metrics: BTreeSet<String>,
    summary: bool,
}

impl<SC: SupervisedLearningComponentsTypes> SupervisedTraining<SC> {
    /// Replace the default training strategy (SingleDeviceTrainingStrategy) with the provided ones.
    ///
    /// # Arguments
    ///
    /// * `training_strategy` - The training strategy.
    pub fn with_training_strategy(mut self, training_strategy: TrainingStrategy<SC>) -> Self {
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
    pub fn with_checkpointing_strategy(
        mut self,
        strategy: <SC::PC as ParadigmComponentsTypes>::CheckpointerStrategy,
    ) -> Self
    where
        // CS: CheckpointingStrategy + 'static,
        <SC::PC as ParadigmComponentsTypes>::CheckpointerStrategy: 'static,
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

    /// Register all metrics as numeric for the training and validation set.
    pub fn metrics<Me: MetricRegistrationV2<SC>>(self, metrics: Me) -> Self {
        metrics.register(self)
    }

    /// Register all metrics as numeric for the training and validation set.
    pub fn metrics_text<Me: TextMetricRegistrationV2<SC>>(self, metrics: Me) -> Self {
        metrics.register(self)
    }

    /// Register a training metric.
    pub fn metric_train<Me: Metric + 'static>(mut self, metric: Me) -> Self
    where
        <OutputTrain<SC::LD> as ItemLazy>::ItemSync: Adaptor<Me::Input>,
    {
        self.metrics.register_train_metric(metric);
        self
    }

    /// Register a validation metric.
    pub fn metric_valid<Me: Metric + 'static>(mut self, metric: Me) -> Self
    where
        <OutputValid<SC::LD> as ItemLazy>::ItemSync: Adaptor<Me::Input>,
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
        <OutputTrain<SC::LD> as ItemLazy>::ItemSync: Adaptor<Me::Input>,
    {
        self.summary_metrics.insert(metric.name().to_string());
        self.metrics.register_train_metric_numeric(metric);
        self
    }

    /// Register a [numeric](crate::metric::Numeric) validation [metric](Metric).
    pub fn metric_valid_numeric<Me: Metric + Numeric + 'static>(mut self, metric: Me) -> Self
    where
        <OutputValid<SC::LD> as ItemLazy>::ItemSync: Adaptor<Me::Input>,
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
        FR: FileRecorder<<SC::LC as LearningComponentsTypes>::Backend> + 'static,
        FR: FileRecorder<
                <<SC::LC as LearningComponentsTypes>::Backend as AutodiffBackend>::InnerBackend,
            > + 'static,
        // <SC::Optimizer as Optimizer<SC::Model, SC::Backend>>::Record: 'static,
        // <SC::LrScheduler as LrScheduler>::Record<SC::Backend>: 'static,
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

    // TODO : pub fn training_strategy()
}

/// Struct to minimise parameters passed to [LearningParadigm::learn].
/// These components are used during training.
pub struct TrainingComponents<SC: SupervisedLearningComponentsTypes> {
    /// The total number of epochs
    pub num_epochs: usize,
    /// The epoch number from which to continue the training.
    pub checkpoint: Option<usize>,
    /// A checkpointer used to load and save learner checkpoints.
    pub checkpointer: Option<LearningCheckpointer<SC::LC, SC::PC>>,
    /// Enables gradients accumulation.
    pub grad_accumulation: Option<usize>,
    /// An [Interupter](Interrupter) that allows aborting the training/evaluation process early.
    pub interrupter: Interrupter,
    /// Cloneable reference to an early stopping strategy.
    pub early_stopping: Option<EarlyStoppingStrategyRef>,
    /// An [EventProcessor](LearnerComponentTypesV2::EventProcessor) that processes events happening during training and validation.
    pub event_processor: <SC::PC as ParadigmComponentsTypes>::EventProcessor,
    /// A reference to an [EventStoreClient](EventStoreClient).
    pub event_store: Arc<EventStoreClient>,
    /// Config for creating a summary of the learning
    pub summary: Option<LearnerSummaryConfig>,
}

impl<SC: SupervisedLearningComponentsTypes + Send + 'static> LearningParadigm<SC::LC>
    for SupervisedTraining<SC>
{
    fn run(mut self, learner: Learner<SC::LC>) -> TrainingResult<SC::InnerModel> {
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
            LearningCheckpointer::new(model, optim, scheduler, self.checkpointer_strategy)
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
                let single_device: SingleDevicetrainingStrategy<SC> =
                    SingleDevicetrainingStrategy::new(device);
                single_device.train(
                    // self.learner,
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
                let multi_device = MultiDeviceLearningStrategyV2::new(devices, multi_device_optim);
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
pub trait MetricRegistrationV2<SC: SupervisedLearningComponentsTypes>: Sized {
    /// Register the metrics.
    fn register(self, builder: SupervisedTraining<SC>) -> SupervisedTraining<SC>;
}

/// Trait to fake variadic generics.
pub trait TextMetricRegistrationV2<SC: SupervisedLearningComponentsTypes>: Sized {
    /// Register the metrics.
    fn register(self, builder: SupervisedTraining<SC>) -> SupervisedTraining<SC>;
}

macro_rules! gen_tuple {
    ($($M:ident),*) => {
        impl<$($M,)* SC: SupervisedLearningComponentsTypes> TextMetricRegistrationV2<SC> for ($($M,)*)
        where
            $(<OutputTrain<SC::LD> as ItemLazy>::ItemSync: Adaptor<$M::Input>,)*
            $(<OutputValid<SC::LD> as ItemLazy>::ItemSync: Adaptor<$M::Input>,)*
            $($M: Metric + 'static,)*
        {
            #[allow(non_snake_case)]
            fn register(
                self,
                builder: SupervisedTraining<SC>,
            ) -> SupervisedTraining<SC> {
                let ($($M,)*) = self;
                $(let builder = builder.metric_train($M.clone());)*
                $(let builder = builder.metric_valid($M);)*
                builder
            }
        }

        impl<$($M,)* SC: SupervisedLearningComponentsTypes> MetricRegistrationV2<SC> for ($($M,)*)
        where
            $(<OutputTrain<SC::LD> as ItemLazy>::ItemSync: Adaptor<$M::Input>,)*
            $(<OutputValid<SC::LD> as ItemLazy>::ItemSync: Adaptor<$M::Input>,)*
            $($M: Metric + Numeric + 'static,)*
        {
            #[allow(non_snake_case)]
            fn register(
                self,
                builder: SupervisedTraining<SC>,
            ) -> SupervisedTraining<SC> {
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
