use crate::Learner;
use crate::checkpoint::{
    AsyncCheckpointer, CheckpointingStrategy, ComposedCheckpointingStrategy, FileCheckpointer,
    KeepLastNCheckpoints, MetricCheckpointingStrategy,
};
use crate::components::{OutputTrain, OutputValid};
use crate::learner::EarlyStoppingStrategy;
use crate::learner::base::Interrupter;
use crate::logger::{FileMetricLogger, MetricLogger};
use crate::metric::processor::{
    AsyncProcessorTraining, FullEventProcessorTraining, ItemLazy, MetricsTraining,
};
use crate::metric::store::{Aggregate, Direction, EventStoreClient, LogEventStore, Split};
use crate::metric::{Adaptor, LossMetric, Metric, Numeric};
use crate::renderer::{MetricsRenderer, default_renderer};
use crate::{
    ApplicationLoggerInstaller, EarlyStoppingStrategyRef, FileApplicationLoggerInstaller,
    LearnerSummaryConfig, LearningCheckpointer, LearningComponentsTypes, LearningDataMarker,
    OffPolicyLearningComponentsMarker, OffPolicyLearningComponentsTypes, ParadigmComponentsMarker,
    ParadigmComponentsTypes, TrainBackend, TrainStep, TrainingComponents, TrainingResult,
    ValidStep,
};
use burn_core::module::Module;
use burn_core::record::FileRecorder;
use burn_core::tensor::backend::AutodiffBackend;
use burn_optim::Optimizer;
use burn_optim::lr_scheduler::LrScheduler;
use burn_optim::lr_scheduler::composed::ComposedLrScheduler;
use burn_rl::{Environment, LearningAgent};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Structure to configure and launch supervised learning trainings.
pub struct OffPolicyLearning<SC: OffPolicyLearningComponentsTypes> {
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
    num_episodes: usize,
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
    // Use BTreeSet instead of HashSet for consistent (alphabetical) iteration order
    summary_metrics: BTreeSet<String>,
    summary: bool,
}

impl<LC, TI, TO, VI, VO, E, A>
    OffPolicyLearning<
        OffPolicyLearningComponentsMarker<
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
            E,
            A,
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
    E: Environment,
    A: LearningAgent<TrainBackend<LC>, E>,
{
    /// Creates a new runner for a supervised training.
    ///
    /// # Arguments
    ///
    /// * `directory` - The directory to save the checkpoints.
    /// * `dataloader_train` - The dataloader for the training split.
    /// * `dataloader_valid` - The dataloader for the validation split.
    pub fn new(directory: impl AsRef<Path>) -> Self {
        let directory = directory.as_ref().to_path_buf();
        let experiment_log_file = directory.join("experiment.log");
        Self {
            num_episodes: 1,
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
            summary_metrics: BTreeSet::new(),
            summary: false,
        }
    }
}

impl<SC: OffPolicyLearningComponentsTypes + Send + 'static> OffPolicyLearning<SC> {
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
    pub fn metrics<Me: MetricRegistration<SC>>(self, metrics: Me) -> Self {
        metrics.register(self)
    }

    /// Register all metrics as numeric for the training and validation set.
    pub fn metrics_text<Me: TextMetricRegistration<SC>>(self, metrics: Me) -> Self {
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

    /// The number of episodes to learn over.
    pub fn num_episodes(mut self, num_episodes: usize) -> Self {
        self.num_episodes = num_episodes;
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
    /// [model](Module) and the [scheduler](LrScheduler) to different files.
    pub fn with_file_checkpointer<FR>(mut self, recorder: FR) -> Self
    where
        FR: FileRecorder<<SC::LC as LearningComponentsTypes>::Backend> + 'static,
        FR: FileRecorder<
                <<SC::LC as LearningComponentsTypes>::Backend as AutodiffBackend>::InnerBackend,
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

    /// Run the training with the specified [Learner](Learner) and dataloaders.
    pub fn run<E: Environment, A: LearningAgent<TrainBackend<SC::LC>, E>>(
        mut self,
        learner: Learner<SC::LC>,
    ) -> TrainingResult<SC::InnerModel> {
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
            num_epochs: self.num_episodes,
            grad_accumulation: self.grad_accumulation,
            summary,
        };
    }
}

/// Trait to fake variadic generics.
pub trait MetricRegistration<SC: OffPolicyLearningComponentsTypes>: Sized {
    /// Register the metrics.
    fn register(self, builder: OffPolicyLearning<SC>) -> OffPolicyLearning<SC>;
}

/// Trait to fake variadic generics.
pub trait TextMetricRegistration<SC: OffPolicyLearningComponentsTypes>: Sized {
    /// Register the metrics.
    fn register(self, builder: OffPolicyLearning<SC>) -> OffPolicyLearning<SC>;
}

macro_rules! gen_tuple {
    ($($M:ident),*) => {
        impl<$($M,)* SC: OffPolicyLearningComponentsTypes + Send + 'static> TextMetricRegistration<SC> for ($($M,)*)
        where
            $(<OutputTrain<SC::LD> as ItemLazy>::ItemSync: Adaptor<$M::Input>,)*
            $(<OutputValid<SC::LD> as ItemLazy>::ItemSync: Adaptor<$M::Input>,)*
            $($M: Metric + 'static,)*
        {
            #[allow(non_snake_case)]
            fn register(
                self,
                builder: OffPolicyLearning<SC>,
            ) -> OffPolicyLearning<SC> {
                let ($($M,)*) = self;
                $(let builder = builder.metric_train($M.clone());)*
                $(let builder = builder.metric_valid($M);)*
                builder
            }
        }

        impl<$($M,)* SC: OffPolicyLearningComponentsTypes + Send + 'static> MetricRegistration<SC> for ($($M,)*)
        where
            $(<OutputTrain<SC::LD> as ItemLazy>::ItemSync: Adaptor<$M::Input>,)*
            $(<OutputValid<SC::LD> as ItemLazy>::ItemSync: Adaptor<$M::Input>,)*
            $($M: Metric + Numeric + 'static,)*
        {
            #[allow(non_snake_case)]
            fn register(
                self,
                builder: OffPolicyLearning<SC>,
            ) -> OffPolicyLearning<SC> {
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
