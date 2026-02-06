use std::sync::Arc;

use crate::{
    EpisodeSummary, EvaluationItem, EventProcessorTraining, ItemLazy, LearnerSummary, RLMetrics,
    metric::store::{Event, EventStoreClient, MetricsUpdate},
    renderer::{MetricState, MetricsRenderer, ProgressType, TrainingProgress},
};

/// Event happening during reinforcement learning.
pub enum RLEvent<TS, ES> {
    /// Signal the start of the process (e.g., learning starts).
    Start,
    /// Signal an agent's training step.
    TrainStep(EvaluationItem<TS>),
    /// Signal a timestep of the agent-environment interface.
    TimeStep(EvaluationItem<ES>),
    /// Signal an episode end.
    EpisodeEnd(EvaluationItem<EpisodeSummary>),
    /// Signal the end of the process (e.g., learning ends).
    End(Option<LearnerSummary>),
}

/// Event happening during evaluation of a reinforcement learning's agent.
pub enum AgentEvaluationEvent<T> {
    /// Signal the start of the process (e.g., training start)
    Start,
    /// Signal a timestep of the agent-environment interface.
    TimeStep(EvaluationItem<T>),
    /// Signal an episode end.
    EpisodeEnd(EvaluationItem<EpisodeSummary>),
    /// Signal the end of the process (e.g., training end).
    End,
}

/// An [event processor](EventProcessorTraining) that handles:
///   - Computing and storing metrics in an [event store](crate::metric::store::EventStore).
///   - Render metrics using a [metrics renderer](MetricsRenderer).
#[derive(new)]
pub struct RLEventProcessor<TS: ItemLazy, ES: ItemLazy> {
    metrics: RLMetrics<TS, ES>,
    renderer: Box<dyn MetricsRenderer>,
    store: Arc<EventStoreClient>,
}

impl<TS: ItemLazy, ES: ItemLazy> RLEventProcessor<TS, ES> {
    fn progress_indicators(&self, progress: &TrainingProgress) -> Vec<ProgressType> {
        let indicators = vec![ProgressType::Detailed {
            tag: String::from("Step"),
            progress: progress.global_progress.clone(),
        }];

        indicators
    }

    fn progress_indicators_eval(&self, progress: &TrainingProgress) -> Vec<ProgressType> {
        let indicators = vec![ProgressType::Detailed {
            tag: String::from("Step"),
            progress: progress.global_progress.clone(),
        }];

        indicators
    }
}

impl<TS: ItemLazy, ES: ItemLazy> RLEventProcessor<TS, ES> {
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

impl<TS: ItemLazy, ES: ItemLazy> EventProcessorTraining<RLEvent<TS, ES>, AgentEvaluationEvent<ES>>
    for RLEventProcessor<TS, ES>
{
    fn process_train(&mut self, event: RLEvent<TS, ES>) {
        match event {
            RLEvent::Start => {
                let definitions = self.metrics.metric_definitions();
                self.store
                    .add_event_train(Event::MetricsInit(definitions.clone()));
                definitions
                    .iter()
                    .for_each(|definition| self.renderer.register_metric(definition.clone()));
            }
            RLEvent::TrainStep(item) => {
                let item = item.sync();
                let metadata = (&item).into();

                let update = self.metrics.update_train_step(&item, &metadata);
                self.process_update_train(update);
            }
            RLEvent::TimeStep(item) => {
                let item = item.sync();
                let progress = (&item).into();
                let metadata = (&item).into();

                let update = self.metrics.update_env_step(&item, &metadata);
                self.process_update_train(update);
                let status = self.progress_indicators(&progress);
                self.renderer.render_train(progress, status);
            }
            RLEvent::EpisodeEnd(item) => {
                let item = item.sync();
                let metadata = (&item).into();

                let update = self.metrics.update_episode_end(&item, &metadata);
                self.process_update_train(update);
            }
            RLEvent::End(learner_summary) => {
                self.renderer.on_train_end(learner_summary).ok();
            }
        }
    }

    fn process_valid(&mut self, event: AgentEvaluationEvent<ES>) {
        match event {
            AgentEvaluationEvent::Start => {} // no-op for now
            AgentEvaluationEvent::TimeStep(item) => {
                let item = item.sync();
                let metadata = (&item).into();

                let update = self.metrics.update_env_step_valid(&item, &metadata);
                self.process_update_valid(update);
            }
            AgentEvaluationEvent::EpisodeEnd(item) => {
                let item = item.sync();
                let progress = (&item).into();
                let metadata = (&item).into();

                let update = self.metrics.update_episode_end_valid(&item, &metadata);
                self.process_update_valid(update);
                let status = self.progress_indicators_eval(&progress);
                self.renderer.render_valid(progress, status);
            }
            AgentEvaluationEvent::End => {} // no-op for now
        }
    }

    fn renderer(self) -> Box<dyn MetricsRenderer> {
        self.renderer
    }
}
