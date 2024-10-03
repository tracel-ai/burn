use super::{Event, EventProcessor, Metrics};
use crate::metric::store::EventStoreClient;
use std::rc::Rc;

/// An [event processor](EventProcessor) that handles:
///   - Computing and storing metrics in an [event store](crate::metric::store::EventStore).
#[derive(new)]
pub(crate) struct MinimalEventProcessor<T, V> {
    metrics: Metrics<T, V>,
    store: Rc<EventStoreClient>,
}

impl<T, V> EventProcessor for MinimalEventProcessor<T, V> {
    type ItemTrain = T;
    type ItemValid = V;

    fn process_train(&mut self, event: Event<Self::ItemTrain>) {
        match event {
            Event::ProcessedItem(item) => {
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
        }
    }

    fn process_valid(&mut self, event: Event<Self::ItemValid>) {
        match event {
            Event::ProcessedItem(item) => {
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
        }
    }
}
