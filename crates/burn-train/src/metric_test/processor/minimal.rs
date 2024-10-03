use super::{Event, EventProcessor, Metrics};
use crate::metric_test::store::EventStoreClient;
use std::rc::Rc;

/// An [event processor](EventProcessor) that handles:
///   - Computing and storing metrics in an [event store](crate::metric::store::EventStore).
#[derive(new)]
pub(crate) struct MinimalEventProcessor<T> {
    metrics: Metrics<T>,
    store: Rc<EventStoreClient>,
}

impl<T> EventProcessor for MinimalEventProcessor<T> {
    type ItemTrain = T;

    fn process(&mut self, event: Event<Self::ItemTrain>) {
        match event {
            Event::ProcessedItem(item) => {
                let metadata = (&item).into();

                let update = self.metrics.update_train(&item, &metadata);

                self.store
                    .add_event_train(crate::metric_test::store::Event::MetricsUpdate(update));
            }
            Event::EndEpoch(epoch) => {
                self.metrics.end_epoch_train();
                self.store
                    .add_event_train(crate::metric_test::store::Event::EndEpoch(epoch));
            }
        }
    }
}
