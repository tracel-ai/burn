use super::{Event, EventProcessorTraining, ItemLazy, MetricsTraining};
use crate::metric::processor::{EventProcessorEvaluation, MetricsEvaluation};
use crate::metric::store::EventStoreClient;
use crate::renderer::{MetricState, MetricsRenderer, MetricsRendererEvaluation};
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
    renderer: Box<dyn MetricsRendererEvaluation>,
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
        renderer: Box<dyn MetricsRendererEvaluation>,
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

    fn process_test(&mut self, event: Event<Self::ItemTest>) {
        match event {
            Event::ProcessedItem(item) => {
                let item = item.sync();
                let progress = (&item).into();
                let metadata = (&item).into();

                let update = self.metrics.update_test(&item, &metadata);

                self.store
                    .add_event_test(crate::metric::store::Event::MetricsUpdate(update.clone()));

                update
                    .entries
                    .into_iter()
                    .for_each(|entry| self.renderer.update_test(MetricState::Generic(entry)));

                update
                    .entries_numeric
                    .into_iter()
                    .for_each(|(entry, value)| {
                        self.renderer
                            .update_test(MetricState::Numeric(entry, value))
                    });

                self.renderer.render_test(progress);
            }
            Event::EndEpoch(_) => {
                log::warn!("Changing epoch doesn't make sense during evaluation")
            }
            Event::End => {
                self.renderer.on_test_end().ok();
            }
        }
    }
}

impl<T: ItemLazy, V: ItemLazy> EventProcessorTraining for FullEventProcessorTraining<T, V> {
    type ItemTrain = T;
    type ItemValid = V;

    fn process_train(&mut self, event: Event<Self::ItemTrain>) {
        match event {
            Event::ProcessedItem(item) => {
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
            Event::EndEpoch(epoch) => {
                self.metrics.end_epoch_train();
                self.store
                    .add_event_train(crate::metric::store::Event::EndEpoch(epoch));
            }
            Event::End => {
                self.renderer.on_train_end().ok();
            }
        }
    }

    fn process_valid(&mut self, event: Event<Self::ItemValid>) {
        match event {
            Event::ProcessedItem(item) => {
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
            Event::EndEpoch(epoch) => {
                self.metrics.end_epoch_valid();
                self.store
                    .add_event_valid(crate::metric::store::Event::EndEpoch(epoch));
            }
            Event::End => {} // no-op for now
        }
    }
    fn renderer(self) -> Option<Box<dyn crate::renderer::MetricsRenderer>> {
        Some(self.renderer)
    }
}
