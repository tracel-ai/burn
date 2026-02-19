use crate::checkpoint::{
    AsyncCheckpointer, CheckpointingStrategy, ComposedCheckpointingStrategy, FileCheckpointer,
    KeepLastNCheckpoints, MetricCheckpointingStrategy,
};
use crate::learner::base::Interrupter;
use crate::logger::{FileMetricLogger, MetricLogger};
use crate::metric::store::{Aggregate, Direction, EventStoreClient, LogEventStore, Split};
use crate::metric::{Adaptor, EpisodeLengthMetric, Metric, Numeric};
use crate::renderer::{MetricsRenderer, default_renderer};
use crate::{
    ApplicationLoggerInstaller, AsyncProcessorTraining, FileApplicationLoggerInstaller, ItemLazy,
    LearnerSummaryConfig, OffPolicyConfig, OffPolicyStrategy, RLAgentRecord, RLCheckpointer,
    RLComponents, RLComponentsMarker, RLComponentsTypes, RLEventProcessor, RLMetrics,
    RLPolicyRecord, RLStrategy,
};
use crate::{EpisodeSummary, RLStrategies};
use burn_core::record::FileRecorder;
use burn_core::tensor::backend::AutodiffBackend;
use burn_rl::{Batchable, Environment, EnvironmentInit, Policy, PolicyLearner, SliceAccess};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Structure to configure and launch reinforcement learning trainings.
pub struct RLTraining<RLC: RLComponentsTypes> {
    // Not that complex. Extracting into yet another type would only make it more confusing.
    #[allow(clippy::type_complexity)]
    checkpointers: Option<(
        AsyncCheckpointer<RLPolicyRecord<RLC>, RLC::Backend>,
        AsyncCheckpointer<RLAgentRecord<RLC>, RLC::Backend>,
    )>,
    num_steps: usize,
    checkpoint: Option<usize>,
    directory: PathBuf,
    grad_accumulation: Option<usize>,
    renderer: Option<Box<dyn MetricsRenderer + 'static>>,
    metrics: RLMetrics<RLC::TrainingOutput, RLC::ActionContext>,
    event_store: LogEventStore,
    interrupter: Interrupter,
    tracing_logger: Option<Box<dyn ApplicationLoggerInstaller>>,
    checkpointer_strategy: Box<dyn CheckpointingStrategy>,
    learning_strategy: RLStrategies<RLC>,
    // Use BTreeSet instead of HashSet for consistent (alphabetical) iteration order
    summary_metrics: BTreeSet<String>,
    summary: bool,
    env_initializer: RLC::EnvInit,
}

impl<B, E, EI, A> RLTraining<RLComponentsMarker<B, E, EI, A>>
where
    B: AutodiffBackend,
    E: Environment + 'static,
    EI: EnvironmentInit<E> + Send + 'static,
    A: PolicyLearner<B> + Send + 'static,
    A::TrainContext: ItemLazy + Clone + Send,
    A::InnerPolicy: Policy<B> + Send,
    <A::InnerPolicy as Policy<B>>::Observation: Batchable + Clone + Send,
    <A::InnerPolicy as Policy<B>>::ActionDistribution: Batchable + Clone + Send,
    <A::InnerPolicy as Policy<B>>::Action: Batchable + Clone + Send,
    <A::InnerPolicy as Policy<B>>::ActionContext: ItemLazy + Clone + Send + 'static,
    <A::InnerPolicy as Policy<B>>::PolicyState: Clone + Send,
    E::State: Into<<A::InnerPolicy as Policy<B>>::Observation> + Clone + Send + 'static,
    E::Action: From<<A::InnerPolicy as Policy<B>>::Action>
        + Into<<A::InnerPolicy as Policy<B>>::Action>
        + Clone
        + Send
        + 'static,
{
    /// Creates a new runner for reinforcement learning.
    ///
    /// # Arguments
    ///
    /// * `directory` - The directory to save the checkpoints.
    /// * `env_init` - Specifies how to initialize the environment.
    pub fn new(directory: impl AsRef<Path>, env_initializer: EI) -> Self {
        let directory = directory.as_ref().to_path_buf();
        let experiment_log_file = directory.join("experiment.log");
        Self {
            num_steps: 1,
            checkpoint: None,
            checkpointers: None,
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
                        &EpisodeLengthMetric::new(), // default to evaluations' cumulative reward.
                        Aggregate::Mean,
                        Direction::Lowest,
                        Split::Valid,
                    ))
                    .build(),
            ),
            learning_strategy: RLStrategies::OffPolicyStrategy(OffPolicyConfig::new()),
            summary_metrics: BTreeSet::new(),
            summary: false,
            env_initializer,
        }
    }
}

impl<RLC: RLComponentsTypes + 'static> RLTraining<RLC> {
    /// Replace the default learning strategy (Off Policy learning) with the provided one.
    ///
    /// # Arguments
    ///
    /// * `training_strategy` - The training strategy.
    pub fn with_learning_strategy(mut self, learning_strategy: RLStrategies<RLC>) -> Self {
        self.learning_strategy = learning_strategy;
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

    /// Register numerical metrics for a training step of the agent.
    pub fn metrics_train<Me: TrainMetricRegistration<RLC>>(self, metrics: Me) -> Self {
        metrics.register(self)
    }

    /// Register textual metrics for a training step of the agent.
    pub fn text_metrics_train<Me: TrainTextMetricRegistration<RLC>>(self, metrics: Me) -> Self {
        metrics.register(self)
    }

    /// Register numerical metrics for each action of the agent.
    pub fn metrics_agent<Me: AgentMetricRegistration<RLC>>(self, metrics: Me) -> Self {
        metrics.register(self)
    }

    /// Register textual metrics for each action of the agent.
    pub fn text_metrics_agent<Me: AgentTextMetricRegistration<RLC>>(self, metrics: Me) -> Self {
        metrics.register(self)
    }

    /// Register numerical metrics for a completed episode.
    pub fn metrics_episode<Me: EpisodeMetricRegistration<RLC>>(self, metrics: Me) -> Self {
        metrics.register(self)
    }

    /// Register textual metrics for a completed episode.
    pub fn text_metrics_episode<Me: EpisodeTextMetricRegistration<RLC>>(self, metrics: Me) -> Self {
        metrics.register(self)
    }

    /// Register a textual metric for a training step.
    pub fn text_metric_train<Me: Metric + 'static>(mut self, metric: Me) -> Self
    where
        <RLC::TrainingOutput as ItemLazy>::ItemSync: Adaptor<Me::Input>,
    {
        self.metrics.register_text_metric_train(metric);
        self
    }

    /// Register a [numeric](crate::metric::Numeric) [metric](Metric) for a training step.
    pub fn metric_train<Me>(mut self, metric: Me) -> Self
    where
        Me: Metric + Numeric + 'static,
        <RLC::TrainingOutput as ItemLazy>::ItemSync: Adaptor<Me::Input>,
    {
        self.summary_metrics.insert(metric.name().to_string());
        self.metrics.register_metric_train(metric);
        self
    }

    /// Register a textual metric for each action taken by the agent.
    pub fn text_metric_agent<Me: Metric + 'static>(mut self, metric: Me) -> Self
    where
        <RLC::ActionContext as ItemLazy>::ItemSync: Adaptor<Me::Input>,
    {
        self.metrics.register_text_metric_agent(metric.clone());
        self.metrics.register_text_metric_agent_valid(metric);
        self
    }

    /// Register a [numeric](crate::metric::Numeric) [metric](Metric) for each action taken by the agent.
    pub fn metric_agent<Me>(mut self, metric: Me) -> Self
    where
        Me: Metric + Numeric + 'static,
        <RLC::ActionContext as ItemLazy>::ItemSync: Adaptor<Me::Input>,
    {
        self.summary_metrics.insert(metric.name().to_string());
        self.metrics.register_agent_metric(metric.clone());
        self.metrics.register_agent_metric_valid(metric);
        self
    }

    /// Register a textual metric for a completed episode.
    pub fn text_metric_episode<Me: Metric + 'static>(mut self, metric: Me) -> Self
    where
        EpisodeSummary: Adaptor<Me::Input> + 'static,
    {
        self.metrics.register_text_metric_episode(metric.clone());
        self.metrics.register_text_metric_episode_valid(metric);
        self
    }

    /// Register a [numeric](crate::metric::Numeric) [metric](Metric) for a completed episode.
    pub fn metric_episode<Me>(mut self, metric: Me) -> Self
    where
        Me: Metric + Numeric + 'static,
        EpisodeSummary: Adaptor<Me::Input> + 'static,
    {
        self.summary_metrics.insert(metric.name().to_string());
        self.metrics.register_episode_metric(metric.clone());
        self.metrics.register_episode_metric_valid(metric);
        self
    }

    /// The number of environment steps to train for.
    pub fn num_steps(mut self, num_steps: usize) -> Self {
        self.num_steps = num_steps;
        self
    }

    /// The step from which the training must resume.
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

    /// Register a checkpointer that will save the environment runner's [policy](Policy)
    /// and the [PolicyLearner](PolicyLearner) state to different files.
    pub fn with_file_checkpointer<FR>(mut self, recorder: FR) -> Self
    where
        FR: FileRecorder<RLC::Backend> + 'static,
        FR: FileRecorder<<RLC::Backend as AutodiffBackend>::InnerBackend> + 'static,
    {
        let checkpoint_dir = self.directory.join("checkpoint");
        let checkpointer_policy =
            FileCheckpointer::new(recorder.clone(), &checkpoint_dir, "policy");
        let checkpointer_learning =
            FileCheckpointer::new(recorder.clone(), &checkpoint_dir, "learning-agent");

        self.checkpointers = Some((
            AsyncCheckpointer::new(checkpointer_policy),
            AsyncCheckpointer::new(checkpointer_learning),
        ));

        self
    }

    /// Enable the training summary report.
    ///
    /// The summary will be displayed after `.launch()`, when the renderer is dropped.
    pub fn summary(mut self) -> Self {
        self.summary = true;
        self
    }

    /// Launch the training with the specified [PolicyLearner](PolicyLearner) on the specified environment.
    pub fn launch(mut self, learner_agent: RLC::LearningAgent) -> RLResult<RLC::Policy>
    where
        RLC::PolicyObs: SliceAccess<RLC::Backend>,
        RLC::PolicyAction: SliceAccess<RLC::Backend>,
    {
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

        let checkpointer = self.checkpointers.map(|(policy, learning_agent)| {
            RLCheckpointer::new(policy, learning_agent, self.checkpointer_strategy)
        });

        let summary = if self.summary {
            Some(LearnerSummaryConfig {
                directory: self.directory,
                metrics: self.summary_metrics.into_iter().collect::<Vec<_>>(),
            })
        } else {
            None
        };

        let components = RLComponents::<RLC> {
            checkpoint: self.checkpoint,
            checkpointer,
            interrupter: self.interrupter,
            event_processor,
            event_store,
            num_steps: self.num_steps,
            grad_accumulation: self.grad_accumulation,
            summary,
        };

        match self.learning_strategy {
            RLStrategies::OffPolicyStrategy(config) => {
                let strategy = OffPolicyStrategy::new(config);
                strategy.train(learner_agent, components, self.env_initializer)
            }
            RLStrategies::Custom(strategy) => {
                strategy.train(learner_agent, components, self.env_initializer)
            }
        }
    }
}

/// The result of reinforcement learning, containing the final policy along with the [renderer](MetricsRenderer).
pub struct RLResult<P> {
    /// The learned policy.
    pub policy: P,
    /// The renderer that can be used for follow up training and evaluation.
    pub renderer: Box<dyn MetricsRenderer>,
}

/// Trait to fake variadic generics for train step metrics.
pub trait AgentMetricRegistration<RLC: RLComponentsTypes>: Sized {
    /// Register the metrics.
    fn register(self, builder: RLTraining<RLC>) -> RLTraining<RLC>;
}

/// Trait to fake variadic generics for train step text metrics.
pub trait AgentTextMetricRegistration<RLC: RLComponentsTypes>: Sized {
    /// Register the metrics.
    fn register(self, builder: RLTraining<RLC>) -> RLTraining<RLC>;
}

/// Trait to fake variadic generics for env step metrics.
pub trait TrainMetricRegistration<RLC: RLComponentsTypes>: Sized {
    /// Register the metrics.
    fn register(self, builder: RLTraining<RLC>) -> RLTraining<RLC>;
}

/// Trait to fake variadic generics for env step text metrics.
pub trait TrainTextMetricRegistration<RLC: RLComponentsTypes>: Sized {
    /// Register the metrics.
    fn register(self, builder: RLTraining<RLC>) -> RLTraining<RLC>;
}

/// Trait to fake variadic generics for episode metrics.
pub trait EpisodeMetricRegistration<RLC: RLComponentsTypes>: Sized {
    /// Register the metrics.
    fn register(self, builder: RLTraining<RLC>) -> RLTraining<RLC>;
}

/// Trait to fake variadic generics for episode text metrics.
pub trait EpisodeTextMetricRegistration<RLC: RLComponentsTypes>: Sized {
    /// Register the metrics.
    fn register(self, builder: RLTraining<RLC>) -> RLTraining<RLC>;
}

macro_rules! gen_tuple {
    ($($M:ident),*) => {
        impl<$($M,)* RLC: RLComponentsTypes + 'static> TrainTextMetricRegistration<RLC> for ($($M,)*)
        where
            $(<RLC::TrainingOutput as ItemLazy>::ItemSync: Adaptor<$M::Input>,)*
            $($M: Metric + 'static,)*
        {
            #[allow(non_snake_case)]
            fn register(
                self,
                builder: RLTraining<RLC>,
            ) -> RLTraining<RLC> {
                let ($($M,)*) = self;
                $(let builder = builder.text_metric_train($M.clone());)*
                builder
            }
        }

        impl<$($M,)* RLC: RLComponentsTypes + 'static> TrainMetricRegistration<RLC> for ($($M,)*)
        where
            $(<RLC::TrainingOutput as ItemLazy>::ItemSync: Adaptor<$M::Input>,)*
            $($M: Metric + Numeric + 'static,)*
        {
            #[allow(non_snake_case)]
            fn register(
                self,
                builder: RLTraining<RLC>,
            ) -> RLTraining<RLC> {
                let ($($M,)*) = self;
                $(let builder = builder.metric_train($M.clone());)*
                builder
            }
        }

        impl<$($M,)* RLC: RLComponentsTypes + 'static> AgentTextMetricRegistration<RLC> for ($($M,)*)
        where
            $(<RLC::ActionContext as ItemLazy>::ItemSync: Adaptor<$M::Input>,)*
            $($M: Metric + 'static,)*
        {
            #[allow(non_snake_case)]
            fn register(
                self,
                builder: RLTraining<RLC>,
            ) -> RLTraining<RLC> {
                let ($($M,)*) = self;
                $(let builder = builder.text_metric_agent($M.clone());)*
                builder
            }
        }

        impl<$($M,)* RLC: RLComponentsTypes + 'static> AgentMetricRegistration<RLC> for ($($M,)*)
        where
            $(<RLC::ActionContext as ItemLazy>::ItemSync: Adaptor<$M::Input>,)*
            $($M: Metric + Numeric + 'static,)*
        {
            #[allow(non_snake_case)]
            fn register(
                self,
                builder: RLTraining<RLC>,
            ) -> RLTraining<RLC> {
                let ($($M,)*) = self;
                $(let builder = builder.metric_agent($M.clone());)*
                builder
            }
        }

        impl<$($M,)* RLC: RLComponentsTypes + 'static> EpisodeTextMetricRegistration<RLC> for ($($M,)*)
        where
            $(EpisodeSummary: Adaptor<$M::Input> + 'static,)*
            $($M: Metric + 'static,)*
        {
            #[allow(non_snake_case)]
            fn register(
                self,
                builder: RLTraining<RLC>,
            ) -> RLTraining<RLC> {
                let ($($M,)*) = self;
                $(let builder = builder.text_metric_episode($M.clone());)*
                builder
            }
        }

        impl<$($M,)* RLC: RLComponentsTypes + 'static> EpisodeMetricRegistration<RLC> for ($($M,)*)
        where
            $(EpisodeSummary: Adaptor<$M::Input> + 'static,)*
            $($M: Metric + Numeric + 'static,)*
        {
            #[allow(non_snake_case)]
            fn register(
                self,
                builder: RLTraining<RLC>,
            ) -> RLTraining<RLC> {
                let ($($M,)*) = self;
                $(let builder = builder.metric_episode($M.clone());)*
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
