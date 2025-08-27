use super::{EventProcessorTraining, ItemLazy, LearnerEvent, MetricsTraining};
use crate::metric::processor::{EvaluatorEvent, EventProcessorEvaluation, MetricsEvaluation};
use crate::metric::store::EventStoreClient;
use crate::renderer::{MetricState, MetricsRenderer};
use std::sync::Arc;

/// An [event processor](EventProcessorTraining) that handles:
///   - Computing and storing metrics in an [event store](crate::metric::store::EventStore).
///   - Render metrics using a [metrics renderer](MetricsRenderer).
pub struct FullEventProcessorTraining<T: ItemLazy, V: ItemLazy> {
    metrics: MetricsTraining<T, V>,
    renderer: Box<dyn MetricsRenderer>,
    store: Arc<EventStoreClient>,
}

/// An [event processor](EventProcessorEvaluation) that handles:
///   - Computing and storing metrics in an [event store](crate::metric::store::EventStore).
///   - Render metrics using a [metrics renderer](MetricsRenderer).
pub struct FullEventProcessorEvaluation<T: ItemLazy> {
    metrics: MetricsEvaluation<T>,
    renderer: Box<dyn MetricsRenderer>,
    store: Arc<EventStoreClient>,
}

impl<T: ItemLazy, V: ItemLazy> FullEventProcessorTraining<T, V> {
    pub(crate) fn new(
        metrics: MetricsTraining<T, V>,
        renderer: Box<dyn MetricsRenderer>,
        store: Arc<EventStoreClient>,
    ) -> Self {
        Self {
            metrics,
            renderer,
            store,
        }
    }
}

impl<T: ItemLazy> FullEventProcessorEvaluation<T> {
    pub(crate) fn new(
        metrics: MetricsEvaluation<T>,
        renderer: Box<dyn MetricsRenderer>,
        store: Arc<EventStoreClient>,
    ) -> Self {
        Self {
            metrics,
            renderer,
            store,
        }
    }
}

impl<T: ItemLazy> EventProcessorEvaluation for FullEventProcessorEvaluation<T> {
    type ItemTest = T;

    fn process_test(&mut self, event: EvaluatorEvent<Self::ItemTest>) {
        match event {
            EvaluatorEvent::ProcessedItem(name, item) => {
                let item = item.sync();
                let progress = (&item).into();
                let metadata = (&item).into();

                let mut update = self.metrics.update_test(&item, &metadata);
                update.tag(name.name.clone());

                self.store
                    .add_event_test(crate::metric::store::Event::MetricsUpdate(update.clone()));

                update.entries.into_iter().for_each(|entry| {
                    self.renderer
                        .update_test(name.clone(), MetricState::Generic(entry))
                });

                update
                    .entries_numeric
                    .into_iter()
                    .for_each(|(entry, value)| {
                        self.renderer
                            .update_test(name.clone(), MetricState::Numeric(entry, value))
                    });

                self.renderer.render_test(progress);
            }
            EvaluatorEvent::End => {
                self.renderer.on_test_end().ok();
            }
        }
    }

    fn renderer(self) -> Box<dyn MetricsRenderer> {
        self.renderer
    }
}

impl<T: ItemLazy, V: ItemLazy> EventProcessorTraining for FullEventProcessorTraining<T, V> {
    type ItemTrain = T;
    type ItemValid = V;

    fn process_train(&mut self, event: LearnerEvent<Self::ItemTrain>) {
        match event {
            LearnerEvent::ProcessedItem(item) => {
                let item = item.sync();
                let progress = (&item).into();
                let metadata = (&item).into();

                let update = self.metrics.update_train(&item, &metadata);

                self.store
                    .add_event_train(crate::metric::store::Event::MetricsUpdate(update.clone()));

                update
                    .entries
                    .into_iter()
                    .for_each(|entry| self.renderer.update_train(MetricState::Generic(entry)));

                update
                    .entries_numeric
                    .into_iter()
                    .for_each(|(entry, value)| {
                        self.renderer
                            .update_train(MetricState::Numeric(entry, value))
                    });

                self.renderer.render_train(progress);
            }
            LearnerEvent::EndEpoch(epoch) => {
                self.metrics.end_epoch_train();
                self.store
                    .add_event_train(crate::metric::store::Event::EndEpoch(epoch));
            }
            LearnerEvent::End => {
                self.renderer.on_train_end().ok();
            }
        }
    }

    fn process_valid(&mut self, event: LearnerEvent<Self::ItemValid>) {
        match event {
            LearnerEvent::ProcessedItem(item) => {
                let item = item.sync();
                let progress = (&item).into();
                let metadata = (&item).into();

                let update = self.metrics.update_valid(&item, &metadata);

                self.store
                    .add_event_valid(crate::metric::store::Event::MetricsUpdate(update.clone()));

                update
                    .entries
                    .into_iter()
                    .for_each(|entry| self.renderer.update_valid(MetricState::Generic(entry)));

                update
                    .entries_numeric
                    .into_iter()
                    .for_each(|(entry, value)| {
                        self.renderer
                            .update_valid(MetricState::Numeric(entry, value))
                    });

                self.renderer.render_valid(progress);
            }
            LearnerEvent::EndEpoch(epoch) => {
                self.metrics.end_epoch_valid();
                self.store
                    .add_event_valid(crate::metric::store::Event::EndEpoch(epoch));
            }
            LearnerEvent::End => {} // no-op for now
        }
    }
    fn renderer(self) -> Box<dyn crate::renderer::MetricsRenderer> {
        self.renderer
    }
}
