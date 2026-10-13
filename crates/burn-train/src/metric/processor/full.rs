use super::{EventProcessorError, EventProcessorTraining, ItemLazy, LearnerEvent, MetricsTraining};
use crate::logger::{EvaluationProgressLogger, TrainingProgressLogger};
use crate::metric::MetricMetadata;
use crate::metric::processor::{EvaluatorEvent, EventProcessorEvaluation, MetricsEvaluation};
use crate::metric::store::{EpochSummary, EventStoreClient, MetricsUpdate, Split};
use crate::renderer::{MetricState, MetricsRenderer};
use std::sync::Arc;

/// An [event processor](EventProcessorTraining) that handles:
///   - Computing and storing metrics in an [event store](crate::metric::store::EventStore).
///   - Render metrics using a [metrics renderer](MetricsRenderer).
pub struct FullEventProcessorTraining<T: ItemLazy, V: ItemLazy> {
    metrics: MetricsTraining<T, V>,
    renderer: Box<dyn MetricsRenderer>,
    store: Arc<EventStoreClient>,
    progress_logger: Option<Box<dyn TrainingProgressLogger>>,
    current_epoch: usize,
    total_epochs: usize,
}

/// An [event processor](EventProcessorEvaluation) that handles:
///   - Computing and storing metrics in an [event store](crate::metric::store::EventStore).
///   - Render metrics using a [metrics renderer](MetricsRenderer).
pub struct FullEventProcessorEvaluation<T: ItemLazy> {
    metrics: MetricsEvaluation<T>,
    renderer: Box<dyn MetricsRenderer>,
    store: Arc<EventStoreClient>,
    progress_logger: Option<Box<dyn EvaluationProgressLogger>>,
    total_tests: usize,
    current_test: usize,
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
            progress_logger: None,
            current_epoch: 1,
            total_epochs: 0,
        }
    }

    pub(crate) fn with_progress_logger(mut self, logger: Box<dyn TrainingProgressLogger>) -> Self {
        self.progress_logger = Some(logger);
        self
    }

    fn handle_train_metrics_update(&mut self, update: MetricsUpdate) {
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

    fn handle_valid_metrics_update(&mut self, update: MetricsUpdate) {
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
            progress_logger: None,
            total_tests: 0,
            current_test: 0,
        }
    }

    pub(crate) fn with_progress_logger(
        mut self,
        logger: Box<dyn EvaluationProgressLogger>,
    ) -> Self {
        self.progress_logger = Some(logger);
        self
    }
}

impl<T: ItemLazy> EventProcessorEvaluation for FullEventProcessorEvaluation<T> {
    type ItemTest = T;

    fn process_test(
        &mut self,
        event: EvaluatorEvent<Self::ItemTest>,
    ) -> Result<(), EventProcessorError> {
        let mut failures = Vec::new();
        match event {
            EvaluatorEvent::Start { total_tests } => {
                let definitions = self.metrics.metric_definitions();
                self.store
                    .add_event_train(crate::metric::store::Event::MetricsInit(
                        definitions.clone(),
                    ));
                definitions
                    .iter()
                    .for_each(|definition| self.renderer.register_metric(definition.clone()));
                self.total_tests = total_tests;
                self.current_test = 0;
                if let Some(logger) = &mut self.progress_logger {
                    logger.start_global_progress(total_tests);
                }
                self.renderer.start_global_progress(total_tests);
            }
            EvaluatorEvent::StartTest(name, total_items) => {
                self.current_test += 1;
                self.renderer.start_test(name.as_str(), total_items);
                if let Some(logger) = &mut self.progress_logger {
                    logger.start_test(name.as_str(), total_items);
                }
            }
            EvaluatorEvent::ProcessedItem(name, item) => {
                let item = match item.sync() {
                    Ok(item) => item,
                    Err(error) => {
                        return Err(EventProcessorError::sync(
                            Split::Test(Some(name.name.clone())),
                            error,
                        ));
                    }
                };
                let metadata = (&item).into();

                let (update, failed) = self.metrics.update_test(
                    &item,
                    &metadata,
                    Split::Test(Some(name.name.clone())),
                );
                failures.extend(failed);

                self.store.add_event_test(
                    crate::metric::store::Event::MetricsUpdate(update.clone()),
                    name.name.clone(),
                );

                update.entries.into_iter().for_each(|entry| {
                    self.renderer
                        .update_test(name.clone(), MetricState::Generic(entry))
                });

                update
                    .entries_numeric
                    .into_iter()
                    .for_each(|numeric_update| {
                        self.renderer.update_test(
                            name.clone(),
                            MetricState::Numeric(
                                numeric_update.entry,
                                numeric_update.numeric_entry,
                            ),
                        )
                    });

                if let Some(logger) = &mut self.progress_logger {
                    logger.update_test_progress(item.progress.items_processed);
                    logger.log_event_evaluation("Iteration".to_string());
                }
                self.renderer
                    .update_test_progress(item.progress.items_processed);
                self.renderer.log_event_evaluation("Iteration".to_string());
            }
            EvaluatorEvent::EndTest => {
                if let Some(logger) = &mut self.progress_logger {
                    logger.end_test();
                }
                self.renderer.end_test();
            }
            EvaluatorEvent::End(summary) => {
                if let Some(logger) = &mut self.progress_logger {
                    logger.end_global_progress();
                }
                self.renderer.end_global_progress();
                self.renderer.on_test_end(summary).ok();
            }
        }
        EventProcessorError::from_errors(failures)
    }

    fn renderer(self) -> Box<dyn MetricsRenderer> {
        self.renderer
    }
}

impl<T: ItemLazy, V: ItemLazy> EventProcessorTraining<LearnerEvent<T>, LearnerEvent<V>>
    for FullEventProcessorTraining<T, V>
{
    fn process_train(&mut self, event: LearnerEvent<T>) -> Result<(), EventProcessorError> {
        let mut failures = Vec::new();
        match event {
            LearnerEvent::Start {
                total_epochs,
                starting_epoch,
                label,
            } => {
                self.total_epochs = total_epochs;
                self.current_epoch = 1;
                let definitions = self.metrics.metric_definitions();
                self.store
                    .add_event_train(crate::metric::store::Event::MetricsInit(
                        definitions.clone(),
                    ));
                definitions
                    .iter()
                    .for_each(|definition| self.renderer.register_metric(definition.clone()));
                if let Some(logger) = &mut self.progress_logger {
                    logger.start(total_epochs, starting_epoch, None, label.as_deref());
                }
                self.renderer
                    .start(total_epochs, starting_epoch, None, label.as_deref());
            }
            LearnerEvent::StartSplit {
                epoch_number,
                total_items,
            } => {
                self.store
                    .add_event_train(crate::metric::store::Event::StartSplit(epoch_number));
                self.renderer.start_split(Split::Train.into(), total_items);
                if let Some(logger) = &mut self.progress_logger {
                    logger.start_split(Split::Train.into(), total_items);
                }
            }
            LearnerEvent::ProcessedItem(item) => {
                let item = match item.sync() {
                    Ok(item) => item,
                    Err(error) => return Err(EventProcessorError::sync(Split::Train, error)),
                };
                let metadata = MetricMetadata {
                    progress: item.progress.clone(),
                    iteration: item.iteration,
                    lr: item.lr.clone(),
                };

                let (update, failed) = self.metrics.update_train(&item, &metadata);
                failures.extend(failed);
                self.handle_train_metrics_update(update);

                if let Some(logger) = &mut self.progress_logger {
                    logger.update_split(item.progress.items_processed);
                    logger.log_event_training("Iteration".to_string());
                }
                self.renderer.update_split(item.progress.items_processed);
                self.renderer.log_event_training("Iteration".to_string());
            }
            LearnerEvent::EndSplit(epoch) => {
                let (update, failed) = self.metrics.end_epoch_train();
                self.handle_train_metrics_update(update);
                failures.extend(failed);

                self.store
                    .add_event_train(crate::metric::store::Event::EndEpoch(EpochSummary::new(
                        epoch,
                        Split::Train,
                    )));
                if let Some(logger) = &mut self.progress_logger {
                    logger.end_split();
                }
                self.renderer.end_split();
            }
            LearnerEvent::EndEpoch(epoch) => {
                self.current_epoch = epoch + 1;
                if let Some(logger) = &mut self.progress_logger {
                    logger.update_epoch(epoch);
                }
                self.renderer.update_epoch(epoch)
            }
            LearnerEvent::End(summary) => {
                if let Some(logger) = &mut self.progress_logger {
                    logger.end();
                }
                self.renderer.end();
                self.renderer.on_train_end(summary).ok();
            }
        }
        EventProcessorError::from_errors(failures)
    }

    fn process_valid(&mut self, event: LearnerEvent<V>) -> Result<(), EventProcessorError> {
        let mut failures = Vec::new();
        match event {
            LearnerEvent::Start { .. } => {} // no-op: valid has no separate start event
            LearnerEvent::StartSplit {
                epoch_number,
                total_items,
            } => {
                self.store
                    .add_event_valid(crate::metric::store::Event::StartSplit(epoch_number));
                if let Some(logger) = &mut self.progress_logger {
                    logger.start_split(Split::Valid.into(), total_items);
                }
                self.renderer.start_split(Split::Valid.into(), total_items);
            }
            LearnerEvent::ProcessedItem(item) => {
                let item = match item.sync() {
                    Ok(item) => item,
                    Err(error) => return Err(EventProcessorError::sync(Split::Valid, error)),
                };
                let metadata = MetricMetadata {
                    progress: item.progress.clone(),
                    iteration: item.iteration,
                    lr: item.lr.clone(),
                };

                let (update, failed) = self.metrics.update_valid(&item, &metadata);
                failures.extend(failed);
                self.handle_valid_metrics_update(update);

                if let Some(logger) = &mut self.progress_logger {
                    logger.update_split(item.progress.items_processed);
                    logger.log_event_training("Iteration".to_string());
                }
                self.renderer.update_split(item.progress.items_processed);
                self.renderer.log_event_training("Iteration".to_string());
            }
            LearnerEvent::EndSplit(epoch) => {
                let (update, failed) = self.metrics.end_epoch_valid();
                self.handle_valid_metrics_update(update);
                failures.extend(failed);

                self.store
                    .add_event_valid(crate::metric::store::Event::EndEpoch(EpochSummary::new(
                        epoch,
                        Split::Valid,
                    )));
                if let Some(logger) = &mut self.progress_logger {
                    logger.end_split();
                }
                self.renderer.end_split();
            }
            LearnerEvent::EndEpoch(_) => {} // update_epoch is handled in process_train(EndEpoch)
            LearnerEvent::End(_) => {}      // no-op
        }
        EventProcessorError::from_errors(failures)
    }
    fn renderer(self) -> Box<dyn MetricsRenderer> {
        self.renderer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TrainingItem;
    use crate::metric::processor::EventProcessorFailure;
    use crate::metric::store::LogEventStore;
    use crate::renderer::cli::CliMetricsRenderer;
    use burn_core::data::dataloader::Progress;
    use burn_std::ExecutionError;

    /// An item whose sync can fail.
    struct Item {
        fails: bool,
    }

    impl ItemLazy for Item {
        fn sync(self) -> Result<Self, ExecutionError> {
            match self.fails {
                true => Err(ExecutionError::with_context("the sync failed")),
                false => Ok(self),
            }
        }
    }

    fn processed(fails: bool) -> LearnerEvent<Item> {
        LearnerEvent::ProcessedItem(TrainingItem::new(
            Item { fails },
            Progress::new(1, 1, None),
            Some(1),
            None,
        ))
    }

    #[test]
    fn a_failed_sync_is_reported_as_one_sync_failure() {
        let mut processor = FullEventProcessorTraining::new(
            MetricsTraining::<Item, Item>::default(),
            Box::new(CliMetricsRenderer::new()),
            Arc::new(EventStoreClient::new(LogEventStore::default())),
        );

        let error = processor.process_train(processed(true)).unwrap_err();
        match error.failures() {
            [EventProcessorFailure::Sync { split, .. }] => assert_eq!(*split, Split::Train),
            other => panic!("expected one sync failure, got {other:?}"),
        }

        // The processor keeps working once items sync again.
        processor.process_train(processed(false)).unwrap();
    }
}
