use super::{EventProcessorTraining, ItemLazy, LearnerEvent, MetricsTraining};
use crate::{metric::store::EventStoreClient, renderer::cli::CliMetricsRenderer};
use std::sync::Arc;

/// An [event processor](EventProcessor) that handles:
///   - Computing and storing metrics in an [event store](crate::metric::store::EventStore).
#[allow(dead_code)]
#[derive(new)]
pub(crate) struct MinimalEventProcessor<T: ItemLazy, V: ItemLazy> {
    metrics: MetricsTraining<T, V>,
    store: Arc<EventStoreClient>,
}

impl<T: ItemLazy, V: ItemLazy> EventProcessorTraining for MinimalEventProcessor<T, V> {
    type ItemTrain = T;
    type ItemValid = V;

    fn process_train(&mut self, event: LearnerEvent<Self::ItemTrain>) {
        match event {
            LearnerEvent::ProcessedItem(item) => {
                let item = item.sync();
                let metadata = (&item).into();

                let update = self.metrics.update_train(&item, &metadata);

                self.store
                    .add_event_train(crate::metric::store::Event::MetricsUpdate(update));
            }
            LearnerEvent::EndEpoch(epoch) => {
                self.metrics.end_epoch_train();
                self.store
                    .add_event_train(crate::metric::store::Event::EndEpoch(epoch));
            }
            LearnerEvent::End => {} // no-op for now
        }
    }

    fn process_valid(&mut self, event: LearnerEvent<Self::ItemValid>) {
        match event {
            LearnerEvent::ProcessedItem(item) => {
                let item = item.sync();
                let metadata = (&item).into();

                let update = self.metrics.update_valid(&item, &metadata);

                self.store
                    .add_event_valid(crate::metric::store::Event::MetricsUpdate(update));
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
        // TODO: Check for another default.
        Box::new(CliMetricsRenderer::new())
    }
}
