use super::{Event, EventProcessor, ItemLazy, Metrics};
use crate::metric::store::EventStoreClient;
use crate::renderer::{MetricState, MetricsRenderer};
use std::sync::Arc;

/// An [event processor](EventProcessor) that handles:
///   - Computing and storing metrics in an [event store](crate::metric::store::EventStore).
///   - Render metrics using a [metrics renderer](MetricsRenderer).
pub struct FullEventProcessor<T: ItemLazy, V: ItemLazy> {
    metrics: Metrics<T, V>,
    renderer: Box<dyn MetricsRenderer>,
    store: Arc<EventStoreClient>,
}

impl<T: ItemLazy, V: ItemLazy> FullEventProcessor<T, V> {
    pub(crate) fn new(
        metrics: Metrics<T, V>,
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

impl<T: ItemLazy, V: ItemLazy> EventProcessor for FullEventProcessor<T, V> {
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
}
