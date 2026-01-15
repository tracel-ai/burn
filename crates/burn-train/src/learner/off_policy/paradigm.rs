use crate::checkpoint::{
    AsyncCheckpointer, CheckpointingStrategy, ComposedCheckpointingStrategy, FileCheckpointer,
    KeepLastNCheckpoints, MetricCheckpointingStrategy,
};
use crate::learner::EarlyStoppingStrategy;
use crate::learner::base::Interrupter;
use crate::logger::{FileMetricLogger, MetricLogger};
use crate::metric::store::{Aggregate, Direction, EventStoreClient, LogEventStore, Split};
use crate::metric::{Adaptor, LossMetric, Metric, Numeric};
use crate::renderer::{MetricsRenderer, default_renderer};
use crate::{
    ApplicationLoggerInstaller, AsyncProcessorTraining, EarlyStoppingStrategyRef,
    FileApplicationLoggerInstaller, ItemLazy, LearnerOptimizerRecord, LearnerSchedulerRecord,
    LearnerSummaryConfig, LearningCheckpointer, LearningComponentsTypes, RLComponents,
    RLEventProcessor, RLMetrics, ReinforcementLearningComponentsMarker,
    ReinforcementLearningComponentsTypes, ReinforcementLearningStrategy, SimpleOffPolicyStrategy,
    TrainingBackend,
};
use crate::{EpisodeSummary, LearnerModelRecord};
use burn_core::record::FileRecorder;
use burn_core::tensor::backend::AutodiffBackend;
use burn_rl::{Environment, LearnerAgent, Policy};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Structure to configure and launch supervised learning trainings.
pub struct OffPolicyLearning<OC: ReinforcementLearningComponentsTypes> {
    // Not that complex. Extracting into another type would only make it more confusing.
    #[allow(clippy::type_complexity)]
    // checkpointers: Option<(
    //     AsyncCheckpointer<LearnerModelRecord<OC::LC>, TrainingBackend<OC::LC>>,
    //     AsyncCheckpointer<LearnerOptimizerRecord<OC::LC>, TrainingBackend<OC::LC>>,
    //     AsyncCheckpointer<LearnerSchedulerRecord<OC::LC>, TrainingBackend<OC::LC>>,
    // )>,
    num_episodes: usize,
    checkpoint: Option<usize>,
    directory: PathBuf,
    grad_accumulation: Option<usize>,
    renderer: Option<Box<dyn MetricsRenderer + 'static>>,
    metrics: RLMetrics<OC::TrainingOutput, OC::ActionContext>,
    event_store: LogEventStore,
    interrupter: Interrupter,
    tracing_logger: Option<Box<dyn ApplicationLoggerInstaller>>,
    checkpointer_strategy: Box<dyn CheckpointingStrategy>,
    early_stopping: Option<EarlyStoppingStrategyRef>,
    // Use BTreeSet instead of HashSet for consistent (alphabetical) iteration order
    summary_metrics: BTreeSet<String>,
    summary: bool,
}

// impl<LC, E, A, TO, AC> OffPolicyLearning<OffPolicyLearningComponentsMarker<LC, E, A, TO, AC>>
// where
//     LC: LearningComponentsTypes,
//     E: Environment + 'static,
//     A: LearnerAgent<TrainingBackend<LC>, E, TrainingOutput = TO, DecisionContext = AC>
//         + Send
//         + 'static,
//     AC: ItemLazy + Clone + Send,
//     TO: ItemLazy + Clone + Send,
impl<B, E, A, TO, AC> OffPolicyLearning<ReinforcementLearningComponentsMarker<B, E, A, TO, AC>>
where
    B: AutodiffBackend,
    E: Environment + 'static,
    A: LearnerAgent<B, E::State, E::Action, TrainingOutput = TO> + Send + 'static,
    A::InnerPolicy: Policy<B, E::State, E::Action, ActionContext = AC> + Send,
    <A::InnerPolicy as Policy<B, E::State, E::Action>>::PolicyState: Send,
    TO: ItemLazy + Clone + Send,
    AC: ItemLazy + Clone + Send + 'static,
{
    /// Creates a new runner for a supervised training.
    ///
    /// # Arguments
    ///
    /// * `directory` - The directory to save the checkpoints.
    pub fn new(directory: impl AsRef<Path>) -> Self {
        let directory = directory.as_ref().to_path_buf();
        let experiment_log_file = directory.join("experiment.log");
        Self {
            num_episodes: 1,
            checkpoint: None,
            // checkpointers: None,
            directory,
            grad_accumulation: None,
            metrics: RLMetrics::default(),
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
                        // &LossMetric::<LC::Backend>::new(), // default to valid loss
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
        }
    }
}

impl<OC: ReinforcementLearningComponentsTypes + 'static> OffPolicyLearning<OC> {
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
    pub fn train_step_metrics<Me: TrainStepMetricRegistration<OC>>(self, metrics: Me) -> Self {
        metrics.register(self)
    }

    /// Register all metrics as numeric for the training and validation set.
    pub fn train_step_metrics_text<Me: TrainStepTextMetricRegistration<OC>>(
        self,
        metrics: Me,
    ) -> Self {
        metrics.register(self)
    }

    /// Register all metrics as numeric for the training and validation set.
    pub fn env_step_metrics<Me: EnvStepMetricRegistration<OC>>(self, metrics: Me) -> Self {
        metrics.register(self)
    }

    /// Register all metrics as numeric for the training and validation set.
    pub fn env_step_metrics_text<Me: EnvStepTextMetricRegistration<OC>>(self, metrics: Me) -> Self {
        metrics.register(self)
    }

    /// Register all metrics as numeric for the training and validation set.
    pub fn episode_metrics<Me: EpisodeMetricRegistration<OC>>(self, metrics: Me) -> Self {
        metrics.register(self)
    }

    /// Register all metrics as numeric for the training and validation set.
    pub fn episode_metrics_text<Me: EpisodeTextMetricRegistration<OC>>(self, metrics: Me) -> Self {
        metrics.register(self)
    }

    /// Register a metric for a step of training.
    pub fn train_step_metric<Me: Metric + 'static>(mut self, metric: Me) -> Self
    where
        <OC::TrainingOutput as ItemLazy>::ItemSync: Adaptor<Me::Input>,
    {
        self.metrics.register_train_step_metric(metric);
        self
    }

    /// Register a [numeric](crate::metric::Numeric) [metric](Metric) for a step of training.
    pub fn train_step_metric_numeric<Me>(mut self, metric: Me) -> Self
    where
        Me: Metric + Numeric + 'static,
        <OC::TrainingOutput as ItemLazy>::ItemSync: Adaptor<Me::Input>,
    {
        self.summary_metrics.insert(metric.name().to_string());
        self.metrics.register_train_step_metric_numeric(metric);
        self
    }

    /// Register a metric for a step of the environment runner.
    pub fn env_step_metric<Me: Metric + 'static>(mut self, metric: Me) -> Self
    where
        <OC::ActionContext as ItemLazy>::ItemSync: Adaptor<Me::Input>,
    {
        self.metrics.register_env_step_metric(metric.clone());
        self.metrics.register_env_step_valid_metric(metric);
        self
    }

    /// Register a [numeric](crate::metric::Numeric) [metric](Metric) for a step of the environment runner.
    pub fn env_step_metric_numeric<Me>(mut self, metric: Me) -> Self
    where
        Me: Metric + Numeric + 'static,
        <OC::ActionContext as ItemLazy>::ItemSync: Adaptor<Me::Input>,
    {
        self.summary_metrics.insert(metric.name().to_string());
        self.metrics
            .register_env_step_metric_numeric(metric.clone());
        self.metrics.register_env_step_valid_metric_numeric(metric);
        self
    }

    /// Register a metric for the end of an episode.
    pub fn episode_metric<Me: Metric + 'static>(mut self, metric: Me) -> Self
    where
        EpisodeSummary: Adaptor<Me::Input> + 'static,
    {
        self.metrics.register_episode_end_metric(metric.clone());
        self.metrics.register_episode_end_valid_metric(metric);
        self
    }

    /// Register a [numeric](crate::metric::Numeric) [metric](Metric) for a step of the environment runner.
    pub fn episode_metric_numeric<Me>(mut self, metric: Me) -> Self
    where
        Me: Metric + Numeric + 'static,
        EpisodeSummary: Adaptor<Me::Input> + 'static,
    {
        self.summary_metrics.insert(metric.name().to_string());
        self.metrics
            .register_episode_end_metric_numeric(metric.clone());
        self.metrics
            .register_episode_end_valid_metric_numeric(metric);
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
    // pub fn with_file_checkpointer<FR>(mut self, recorder: FR) -> Self
    // where
    //     FR: FileRecorder<<OC::LC as LearningComponentsTypes>::Backend> + 'static,
    //     FR: FileRecorder<
    //             <<OC::LC as LearningComponentsTypes>::Backend as AutodiffBackend>::InnerBackend,
    //         > + 'static,
    // {
    //     let checkpoint_dir = self.directory.join("checkpoint");
    //     let checkpointer_model = FileCheckpointer::new(recorder.clone(), &checkpoint_dir, "model");
    //     let checkpointer_optimizer =
    //         FileCheckpointer::new(recorder.clone(), &checkpoint_dir, "optim");
    //     let checkpointer_scheduler: FileCheckpointer<FR> =
    //         FileCheckpointer::new(recorder, &checkpoint_dir, "scheduler");

    //     self.checkpointers = Some((
    //         AsyncCheckpointer::new(checkpointer_model),
    //         AsyncCheckpointer::new(checkpointer_optimizer),
    //         AsyncCheckpointer::new(checkpointer_scheduler),
    //     ));

    //     self
    // }

    /// Enable the training summary report.
    ///
    /// The summary will be displayed after `.fit()`, when the renderer is dropped.
    pub fn summary(mut self) -> Self {
        self.summary = true;
        self
    }

    /// Run the training with the specified [Learner](Learner) and dataloaders.
    pub fn launch(
        mut self,
        learner_agent: OC::LearningAgent,
        // ) -> <OC::LearningAgent as Agent<TrainingBackend<OC::LC>, OC::Env>>::Policy {
    ) -> OC::Policy {
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
        let event_processor = AsyncProcessorTraining::new(RLEventProcessor::new(
            self.metrics,
            renderer,
            event_store.clone(),
        ));

        // let checkpointer = self.checkpointers.map(|(model, optim, scheduler)| {
        //     LearningCheckpointer::new(model, optim, scheduler, self.checkpointer_strategy)
        // });

        let summary = if self.summary {
            Some(LearnerSummaryConfig {
                directory: self.directory,
                metrics: self.summary_metrics.into_iter().collect::<Vec<_>>(),
            })
        } else {
            None
        };

        // TODO: pq on a besoin du type?
        let components = RLComponents::<OC> {
            checkpoint: self.checkpoint,
            checkpointer: None,
            interrupter: self.interrupter,
            early_stopping: self.early_stopping,
            event_processor,
            event_store,
            num_epochs: self.num_episodes,
            grad_accumulation: self.grad_accumulation,
            summary,
        };

        let strategy = SimpleOffPolicyStrategy::new(Default::default());
        strategy.train(learner_agent, components).model
    }
}

/// Trait to fake variadic generics for train step metrics.
pub trait EnvStepMetricRegistration<OC: ReinforcementLearningComponentsTypes>: Sized {
    /// Register the metrics.
    fn register(self, builder: OffPolicyLearning<OC>) -> OffPolicyLearning<OC>;
}

/// Trait to fake variadic generics for train step text metrics.
pub trait EnvStepTextMetricRegistration<OC: ReinforcementLearningComponentsTypes>: Sized {
    /// Register the metrics.
    fn register(self, builder: OffPolicyLearning<OC>) -> OffPolicyLearning<OC>;
}

/// Trait to fake variadic generics for env step metrics.
pub trait TrainStepMetricRegistration<OC: ReinforcementLearningComponentsTypes>: Sized {
    /// Register the metrics.
    fn register(self, builder: OffPolicyLearning<OC>) -> OffPolicyLearning<OC>;
}

/// Trait to fake variadic generics for env step text metrics.
pub trait TrainStepTextMetricRegistration<OC: ReinforcementLearningComponentsTypes>: Sized {
    /// Register the metrics.
    fn register(self, builder: OffPolicyLearning<OC>) -> OffPolicyLearning<OC>;
}

/// Trait to fake variadic generics for episode metrics.
pub trait EpisodeMetricRegistration<OC: ReinforcementLearningComponentsTypes>: Sized {
    /// Register the metrics.
    fn register(self, builder: OffPolicyLearning<OC>) -> OffPolicyLearning<OC>;
}

/// Trait to fake variadic generics for episode text metrics.
pub trait EpisodeTextMetricRegistration<OC: ReinforcementLearningComponentsTypes>: Sized {
    /// Register the metrics.
    fn register(self, builder: OffPolicyLearning<OC>) -> OffPolicyLearning<OC>;
}

macro_rules! gen_tuple {
    ($($M:ident),*) => {
        impl<$($M,)* OC: ReinforcementLearningComponentsTypes + 'static> TrainStepTextMetricRegistration<OC> for ($($M,)*)
        where
            $(<OC::TrainingOutput as ItemLazy>::ItemSync: Adaptor<$M::Input>,)*
            $($M: Metric + 'static,)*
        {
            #[allow(non_snake_case)]
            fn register(
                self,
                builder: OffPolicyLearning<OC>,
            ) -> OffPolicyLearning<OC> {
                let ($($M,)*) = self;
                $(let builder = builder.train_step_metric($M.clone());)*
                builder
            }
        }

        impl<$($M,)* OC: ReinforcementLearningComponentsTypes + 'static> TrainStepMetricRegistration<OC> for ($($M,)*)
        where
            $(<OC::TrainingOutput as ItemLazy>::ItemSync: Adaptor<$M::Input>,)*
            $($M: Metric + Numeric + 'static,)*
        {
            #[allow(non_snake_case)]
            fn register(
                self,
                builder: OffPolicyLearning<OC>,
            ) -> OffPolicyLearning<OC> {
                let ($($M,)*) = self;
                $(let builder = builder.train_step_metric_numeric($M.clone());)*
                builder
            }
        }

        impl<$($M,)* OC: ReinforcementLearningComponentsTypes + 'static> EnvStepTextMetricRegistration<OC> for ($($M,)*)
        where
            $(<OC::ActionContext as ItemLazy>::ItemSync: Adaptor<$M::Input>,)*
            $($M: Metric + 'static,)*
        {
            #[allow(non_snake_case)]
            fn register(
                self,
                builder: OffPolicyLearning<OC>,
            ) -> OffPolicyLearning<OC> {
                let ($($M,)*) = self;
                $(let builder = builder.env_step_metric($M.clone());)*
                builder
            }
        }

        impl<$($M,)* OC: ReinforcementLearningComponentsTypes + 'static> EnvStepMetricRegistration<OC> for ($($M,)*)
        where
            $(<OC::ActionContext as ItemLazy>::ItemSync: Adaptor<$M::Input>,)*
            $($M: Metric + Numeric + 'static,)*
        {
            #[allow(non_snake_case)]
            fn register(
                self,
                builder: OffPolicyLearning<OC>,
            ) -> OffPolicyLearning<OC> {
                let ($($M,)*) = self;
                $(let builder = builder.env_step_metric_numeric($M.clone());)*
                builder
            }
        }

        impl<$($M,)* OC: ReinforcementLearningComponentsTypes + 'static> EpisodeTextMetricRegistration<OC> for ($($M,)*)
        where
            $(EpisodeSummary: Adaptor<$M::Input> + 'static,)*
            $($M: Metric + 'static,)*
        {
            #[allow(non_snake_case)]
            fn register(
                self,
                builder: OffPolicyLearning<OC>,
            ) -> OffPolicyLearning<OC> {
                let ($($M,)*) = self;
                $(let builder = builder.episode_metric($M.clone());)*
                builder
            }
        }

        impl<$($M,)* OC: ReinforcementLearningComponentsTypes + 'static> EpisodeMetricRegistration<OC> for ($($M,)*)
        where
            $(EpisodeSummary: Adaptor<$M::Input> + 'static,)*
            $($M: Metric + Numeric + 'static,)*
        {
            #[allow(non_snake_case)]
            fn register(
                self,
                builder: OffPolicyLearning<OC>,
            ) -> OffPolicyLearning<OC> {
                let ($($M,)*) = self;
                $(let builder = builder.episode_metric_numeric($M.clone());)*
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
