use super::{Event, EventProcessor, ItemLazy, Metrics};
use crate::metric::store::EventStoreClient;
use std::sync::Arc;

/// An [event processor](EventProcessor) that handles:
///   - Computing and storing metrics in an [event store](crate::metric::store::EventStore).
#[derive(new)]
pub(crate) struct MinimalEventProcessor<T: ItemLazy, V: ItemLazy> {
    metrics: Metrics<T, V>,
    store: Arc<EventStoreClient>,
}

impl<T: ItemLazy, V: ItemLazy> EventProcessor for MinimalEventProcessor<T, V> {
    type ItemTrain = T;
    type ItemValid = V;

    fn process_train(&mut self, event: Event<Self::ItemTrain>) {
        match event {
            Event::ProcessedItem(item) => {
                let item = item.sync();
                let metadata = (&item).into();

                let update = self.metrics.update_train(&item, &metadata);

                self.store
                    .add_event_train(crate::metric::store::Event::MetricsUpdate(update));
            }
            Event::EndEpoch(epoch) => {
                self.metrics.end_epoch_train();
                self.store
                    .add_event_train(crate::metric::store::Event::EndEpoch(epoch));
            }
            Event::End => {} // no-op for now
        }
    }

    fn process_valid(&mut self, event: Event<Self::ItemValid>) {
        match event {
            Event::ProcessedItem(item) => {
                let item = item.sync();
                let metadata = (&item).into();

                let update = self.metrics.update_valid(&item, &metadata);

                self.store
                    .add_event_valid(crate::metric::store::Event::MetricsUpdate(update));
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
