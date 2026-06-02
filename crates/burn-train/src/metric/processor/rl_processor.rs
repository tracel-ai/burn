use std::sync::Arc;

use crate::{
    EvaluationItem, EventProcessorTraining, LearnerSummary, RLMetrics,
    logger::TrainingProgressLogger,
    metric::store::{Event, EventStoreClient, MetricsUpdate},
    renderer::{MetricState, MetricsRenderer},
};

/// Event happening during reinforcement learning.
///
/// The carried items are type-erased (see [`EvaluationItem::new`]) so this event
/// and its processor are not generic over the agent's output types.
pub enum RLEvent {
    /// Signal the start of the process (e.g., learning starts).
    Start {
        /// The total number of items to process during training (e.g., total number of environment steps).
        total_items: usize,
    },
    /// Signal an agent's training step.
    TrainStep(EvaluationItem),
    /// Signal a timestep of the agent-environment interface.
    EnvStep(EvaluationItem),
    /// Signal an episode end.
    EpisodeEnd(EvaluationItem),
    /// Signal the end of the process (e.g., learning ends).
    End(Option<LearnerSummary>),
}

/// Event happening during evaluation of a reinforcement learning's agent.
pub enum AgentEvaluationEvent {
    /// Signal the start of the process (e.g., training start)
    Start(usize),
    /// Signal a timestep of the agent-environment interface.
    EnvStep(EvaluationItem),
    /// Signal an episode end.
    EpisodeEnd(EvaluationItem),
    /// Signal the end of the process (e.g., training end).
    End,
}

/// An [event processor](EventProcessorTraining) that handles:
///   - Computing and storing metrics in an [event store](crate::metric::store::EventStore).
///   - Render metrics using a [metrics renderer](MetricsRenderer).
pub struct RLEventProcessor {
    metrics: RLMetrics,
    renderer: Box<dyn MetricsRenderer>,
    store: Arc<EventStoreClient>,
    training_progress_logger: Option<Box<dyn TrainingProgressLogger>>,
}

impl RLEventProcessor {
    pub(crate) fn new(
        metrics: RLMetrics,
        renderer: Box<dyn MetricsRenderer>,
        store: Arc<EventStoreClient>,
    ) -> Self {
        Self {
            metrics,
            renderer,
            store,
            training_progress_logger: None,
        }
    }

    fn process_update_train(&mut self, update: MetricsUpdate) {
        self.store
            .add_event_train(crate::metric::store::Event::MetricsUpdate(update.clone()));

        update
            .entries
            .into_iter()
            .for_each(|entry| self.renderer.update_train(MetricState::Generic(entry)));

        update
            .entries_numeric
            .into_iter()
            .for_each(|numeric_update| {
                self.renderer.update_train(MetricState::Numeric(
                    numeric_update.entry,
                    numeric_update.numeric_entry,
                ))
            });
    }

    fn process_update_valid(&mut self, update: MetricsUpdate) {
        self.store
            .add_event_valid(crate::metric::store::Event::MetricsUpdate(update.clone()));

        update
            .entries
            .into_iter()
            .for_each(|entry| self.renderer.update_valid(MetricState::Generic(entry)));

        update
            .entries_numeric
            .into_iter()
            .for_each(|numeric_update| {
                self.renderer.update_valid(MetricState::Numeric(
                    numeric_update.entry,
                    numeric_update.numeric_entry,
                ))
            });
    }
}

impl EventProcessorTraining<RLEvent, AgentEvaluationEvent> for RLEventProcessor {
    fn process_train(&mut self, event: RLEvent) {
        match event {
            RLEvent::Start { total_items } => {
                let definitions = self.metrics.metric_definitions();
                self.store
                    .add_event_train(Event::MetricsInit(definitions.clone()));
                definitions
                    .iter()
                    .for_each(|definition| self.renderer.register_metric(definition.clone()));
                if let Some(logger) = &mut self.training_progress_logger {
                    logger.start(0, Some(total_items));
                }
                self.renderer.start(0, Some(total_items));
            }
            RLEvent::TrainStep(item) => {
                let item = item.sync();
                let metadata = (&item).into();

                let update = self.metrics.update_train_step(&item, &metadata);
                self.process_update_train(update);

                if let Some(logger) = &mut self.training_progress_logger {
                    logger.log_event_training("TrainStep".to_string());
                }
                self.renderer.log_event_training("TrainStep".to_string());
            }
            RLEvent::EnvStep(item) => {
                let item = item.sync();
                let metadata = (&item).into();

                let update = self.metrics.update_env_step(&item, &metadata);
                self.process_update_train(update);

                if let Some(logger) = &mut self.training_progress_logger {
                    logger.update_split(item.progress.items_processed);
                    logger.log_event_training("EnvStep".to_string());
                }
                self.renderer.update_split(item.progress.items_processed);
                self.renderer.log_event_training("EnvStep".to_string());
            }
            RLEvent::EpisodeEnd(item) => {
                let item = item.sync();
                let metadata = (&item).into();

                let update = self.metrics.update_episode_end(&item, &metadata);
                self.process_update_train(update);

                if let Some(logger) = &mut self.training_progress_logger {
                    logger.log_event_training("EpisodeEnd".to_string());
                }
                self.renderer.log_event_training("EpisodeEnd".to_string());
            }
            RLEvent::End(learner_summary) => {
                if let Some(logger) = &mut self.training_progress_logger {
                    logger.end();
                }
                self.renderer.end();
                self.renderer.on_train_end(learner_summary).ok();
            }
        }
    }

    fn process_valid(&mut self, event: AgentEvaluationEvent) {
        match event {
            AgentEvaluationEvent::Start(num_episodes) => {
                if let Some(logger) = &mut self.training_progress_logger {
                    logger.start_split("valid", num_episodes);
                }
                self.renderer.start_split("valid", num_episodes);
            }
            AgentEvaluationEvent::EnvStep(item) => {
                let item = item.sync();
                let metadata = (&item).into();

                let update = self.metrics.update_env_step_valid(&item, &metadata);
                self.process_update_valid(update);

                if let Some(logger) = &mut self.training_progress_logger {
                    logger.log_event_training("EnvStep".to_string());
                }
                self.renderer.log_event_training("EnvStep".to_string());
            }
            AgentEvaluationEvent::EpisodeEnd(item) => {
                let item = item.sync();
                let metadata = (&item).into();

                let update = self.metrics.update_episode_end_valid(&item, &metadata);
                self.process_update_valid(update);

                if let Some(logger) = &mut self.training_progress_logger {
                    logger.update_split(item.progress.items_processed);
                    logger.log_event_training("EpisodeEnd".to_string());
                }
                self.renderer.update_split(item.progress.items_processed);
                self.renderer.log_event_training("EpisodeEnd".to_string());
            }
            AgentEvaluationEvent::End => {
                if let Some(logger) = &mut self.training_progress_logger {
                    logger.end_split();
                }
                self.renderer.end_split();
            }
        }
    }

    fn renderer(self) -> Box<dyn MetricsRenderer> {
        self.renderer
    }
}
