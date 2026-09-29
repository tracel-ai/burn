use super::{EventProcessorTraining, ItemLazy, LearnerEvent, MetricsError, MetricsTraining};
use crate::{
    logger::TrainingProgressLogger,
    metric::store::{EpochSummary, EventStoreClient, Split},
    renderer::cli::CliMetricsRenderer,
};
use std::sync::Arc;

/// An [event processor](EventProcessor) that handles:
///   - Computing and storing metrics in an [event store](crate::metric::store::EventStore).
///   - Optionally logging training progress via a [TrainingProgressLogger].
#[allow(dead_code)]
pub(crate) struct MinimalEventProcessor<T: ItemLazy, V: ItemLazy> {
    metrics: MetricsTraining<T, V>,
    store: Arc<EventStoreClient>,
    progress_logger: Option<Box<dyn TrainingProgressLogger>>,
}

#[allow(dead_code)]
impl<T: ItemLazy, V: ItemLazy> MinimalEventProcessor<T, V> {
    pub(crate) fn new(metrics: MetricsTraining<T, V>, store: Arc<EventStoreClient>) -> Self {
        Self {
            metrics,
            store,
            progress_logger: None,
        }
    }

    pub(crate) fn with_progress_logger(mut self, logger: Box<dyn TrainingProgressLogger>) -> Self {
        self.progress_logger = Some(logger);
        self
    }
}

impl<T: ItemLazy, V: ItemLazy> EventProcessorTraining<LearnerEvent<T>, LearnerEvent<V>>
    for MinimalEventProcessor<T, V>
{
    fn process_train(&mut self, event: LearnerEvent<T>) -> Result<(), MetricsError> {
        let mut failures = Vec::new();
        match event {
            LearnerEvent::Start {
                total_epochs,
                starting_epoch,
                label,
            } => {
                let definitions = self.metrics.metric_definitions();
                self.store
                    .add_event_train(crate::metric::store::Event::MetricsInit(definitions));
                if let Some(logger) = &mut self.progress_logger {
                    logger.start(total_epochs, starting_epoch, None, label.as_deref());
                }
            }
            LearnerEvent::StartSplit {
                epoch_number,
                total_items,
            } => {
                self.store
                    .add_event_train(crate::metric::store::Event::StartSplit(epoch_number));
                if let Some(logger) = &mut self.progress_logger {
                    logger.start_split(Split::Train.into(), total_items);
                }
            }
            LearnerEvent::ProcessedItem(item) => {
                let item = item.sync();
                let metadata = (&item).into();

                let (update, failed) = self.metrics.update_train(&item, &metadata);
                failures.extend(failed);
                self.store
                    .add_event_train(crate::metric::store::Event::MetricsUpdate(update));

                if let Some(logger) = &mut self.progress_logger {
                    logger.update_split(item.progress.items_processed);
                }
            }
            LearnerEvent::EndSplit(epoch) => {
                let (update, failed) = self.metrics.end_epoch_train();
                self.store
                    .add_event_train(crate::metric::store::Event::MetricsUpdate(update));
                failures.extend(failed);

                self.store
                    .add_event_train(crate::metric::store::Event::EndEpoch(EpochSummary::new(
                        epoch,
                        Split::Train,
                    )));
                if let Some(logger) = &mut self.progress_logger {
                    logger.end_split();
                }
            }
            LearnerEvent::EndEpoch(epoch) => {
                if let Some(logger) = &mut self.progress_logger {
                    logger.update_epoch(epoch);
                }
            }
            LearnerEvent::End(_summary) => {
                if let Some(logger) = &mut self.progress_logger {
                    logger.end();
                }
            }
        }
        MetricsError::from_errors(failures)
    }

    fn process_valid(&mut self, event: LearnerEvent<V>) -> Result<(), MetricsError> {
        let mut failures = Vec::new();
        match event {
            LearnerEvent::Start { .. } => {} // no-op
            LearnerEvent::StartSplit {
                epoch_number,
                total_items,
            } => {
                self.store
                    .add_event_valid(crate::metric::store::Event::StartSplit(epoch_number));
                if let Some(logger) = &mut self.progress_logger {
                    logger.start_split(Split::Valid.into(), total_items);
                }
            }
            LearnerEvent::ProcessedItem(item) => {
                let item = item.sync();
                let metadata = (&item).into();

                let (update, failed) = self.metrics.update_valid(&item, &metadata);
                failures.extend(failed);
                self.store
                    .add_event_valid(crate::metric::store::Event::MetricsUpdate(update));

                if let Some(logger) = &mut self.progress_logger {
                    logger.update_split(item.progress.items_processed);
                }
            }
            LearnerEvent::EndSplit(epoch) => {
                let (update, failed) = self.metrics.end_epoch_valid();
                self.store
                    .add_event_valid(crate::metric::store::Event::MetricsUpdate(update));
                failures.extend(failed);

                self.store
                    .add_event_valid(crate::metric::store::Event::EndEpoch(EpochSummary::new(
                        epoch,
                        Split::Valid,
                    )));
                if let Some(logger) = &mut self.progress_logger {
                    logger.end_split();
                }
            }
            LearnerEvent::EndEpoch(_) => {} // update_epoch handled in process_train(EndEpoch)
            LearnerEvent::End(_) => {}      // no-op: End is only emitted on process_train
        }
        MetricsError::from_errors(failures)
    }
    fn renderer(self) -> Box<dyn crate::renderer::MetricsRenderer> {
        // TODO: Check for another default.
        Box::new(CliMetricsRenderer::new())
    }
}
