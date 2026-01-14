use std::sync::Arc;

use crate::{
    ItemLazy,
    metric::{
        rl_processor::{
            RlEvaluationEvent, RlEventProcessorTrain, RlTrainingEvent, metrics::RlMetricsTraining,
        },
        store::{Event, EventStoreClient, MetricsUpdate},
    },
    renderer::{MetricState, MetricsRenderer, TrainingProgress},
};

/// An [event processor](EventProcessorTraining) that handles:
///   - Computing and storing metrics in an [event store](crate::metric::store::EventStore).
///   - Render metrics using a [metrics renderer](MetricsRenderer).
#[derive(new)]
pub struct FullEventProcessorTrainingRl<TS: ItemLazy, ES: ItemLazy> {
    metrics: RlMetricsTraining<TS, ES>,
    renderer: Box<dyn MetricsRenderer>,
    store: Arc<EventStoreClient>,
}

// /// An [event processor](EventProcessorEvaluation) that handles:
// ///   - Computing and storing metrics in an [event store](crate::metric::store::EventStore).
// ///   - Render metrics using a [metrics renderer](MetricsRenderer).
// pub struct FullEventProcessorEvaluationRl<TS: ItemLazy> {
//     metrics: MetricsEvaluation<TS>,
//     renderer: Box<dyn MetricsRenderer>,
//     store: Arc<EventStoreClient>,
// }

impl<TS: ItemLazy, ES: ItemLazy> FullEventProcessorTrainingRl<TS, ES> {
    fn process_update_train(&mut self, update: MetricsUpdate, progress: TrainingProgress) {
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

        self.renderer.render_train(progress);
    }

    fn process_update_valid(&mut self, update: MetricsUpdate, progress: TrainingProgress) {
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

        self.renderer.render_valid(progress);
    }
}
impl<TS: ItemLazy, ES: ItemLazy> RlEventProcessorTrain for FullEventProcessorTrainingRl<TS, ES> {
    type TrainingOutput = TS;
    type ActionContext = ES;

    fn process_train(&mut self, event: RlTrainingEvent<Self::TrainingOutput, Self::ActionContext>) {
        match event {
            RlTrainingEvent::Start => {
                let definitions = self.metrics.metric_definitions();
                self.store
                    .add_event_train(Event::MetricsInit(definitions.clone()));
                definitions
                    .iter()
                    .for_each(|definition| self.renderer.register_metric(definition.clone()));
            }
            RlTrainingEvent::TrainStep(item) => {
                let item = item.sync();
                let progress = (&item).into();
                let metadata = (&item).into();

                let update = self.metrics.update_train_step(&item, &metadata);
                self.process_update_train(update, progress);
            }
            RlTrainingEvent::EnvStep(item) => {
                let item = item.sync();
                let progress = (&item).into();
                let metadata = (&item).into();

                let update = self.metrics.update_env_step(&item, &metadata);
                self.process_update_train(update, progress);
            }
            RlTrainingEvent::EpisodeEnd(item) => {
                let item = item.sync();
                let progress = (&item).into();
                let metadata = (&item).into();

                let update = self.metrics.update_episode_end(&item, &metadata);
                self.process_update_train(update, progress);
            }
            RlTrainingEvent::End(learner_summary) => {
                self.renderer.on_train_end(learner_summary).ok();
            }
        }
    }

    fn process_valid(&mut self, event: RlEvaluationEvent<Self::ActionContext>) {
        match event {
            RlEvaluationEvent::Start => {} // no-op for now
            RlEvaluationEvent::EnvStep(item) => {
                let item = item.sync();
                let progress = (&item).into();
                let metadata = (&item).into();

                let update = self.metrics.update_env_step_valid(&item, &metadata);
                self.process_update_valid(update, progress);
            }
            RlEvaluationEvent::EpisodeEnd(item) => {
                let item = item.sync();
                let progress = (&item).into();
                let metadata = (&item).into();

                let update = self.metrics.update_episode_end_valid(&item, &metadata);
                self.process_update_valid(update, progress);
            }
            RlEvaluationEvent::End => {} // no-op for now
        }
    }

    fn renderer(self) -> Box<dyn MetricsRenderer> {
        todo!()
    }
}
