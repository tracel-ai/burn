use super::{EventProcessorTraining, ItemLazy, LearnerEvent, MetricsTraining};
use crate::logger::{EvaluationProgressLogger, TrainingProgressLogger};
use crate::metric::MetricMetadata;
use crate::metric::processor::{EvaluatorEvent, EventProcessorEvaluation, MetricsEvaluation};
use crate::metric::store::{EpochSummary, EventStoreClient, Split};
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

    fn process_test(&mut self, event: EvaluatorEvent<Self::ItemTest>) {
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
                let item = item.sync();
                let metadata = (&item).into();

                let update = self.metrics.update_test(&item, &metadata);

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
    }

    fn renderer(self) -> Box<dyn MetricsRenderer> {
        self.renderer
    }
}

impl<T: ItemLazy, V: ItemLazy> EventProcessorTraining<LearnerEvent<T>, LearnerEvent<V>>
    for FullEventProcessorTraining<T, V>
{
    fn process_train(&mut self, event: LearnerEvent<T>) {
        match event {
            LearnerEvent::Start { total_epochs } => {
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
                    logger.start(total_epochs, None);
                }
                self.renderer.start(total_epochs, None);
            }
            LearnerEvent::StartSplit(total_items) => {
                self.renderer.start_split("train", total_items);
                if let Some(logger) = &mut self.progress_logger {
                    logger.start_split("train", total_items);
                }
            }
            LearnerEvent::ProcessedItem(item) => {
                let item = item.sync();
                let metadata = MetricMetadata {
                    progress: item.progress.clone(),
                    iteration: item.iteration,
                    lr: item.lr,
                };

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
                    .for_each(|numeric_update| {
                        self.renderer.update_train(MetricState::Numeric(
                            numeric_update.entry,
                            numeric_update.numeric_entry,
                        ))
                    });

                if let Some(logger) = &mut self.progress_logger {
                    logger.update_split(item.progress.items_processed);
                    logger.log_event_training("Iteration".to_string());
                }
                self.renderer.update_split(item.progress.items_processed);
                self.renderer.log_event_training("Iteration".to_string());
            }
            LearnerEvent::EndSplit(epoch) => {
                self.store
                    .add_event_train(crate::metric::store::Event::EndEpoch(EpochSummary::new(
                        epoch,
                        Split::Train,
                    )));
                if let Some(logger) = &mut self.progress_logger {
                    logger.end_split();
                }
                self.renderer.end_split();
                self.metrics.end_epoch_train();
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
    }

    fn process_valid(&mut self, event: LearnerEvent<V>) {
        match event {
            LearnerEvent::Start { .. } => {} // no-op: valid has no separate start event
            LearnerEvent::StartSplit(total_items) => {
                if let Some(logger) = &mut self.progress_logger {
                    logger.start_split("valid", total_items);
                }
                self.renderer.start_split("valid", total_items);
            }
            LearnerEvent::ProcessedItem(item) => {
                let item = item.sync();
                let metadata = MetricMetadata {
                    progress: item.progress.clone(),
                    iteration: item.iteration,
                    lr: item.lr,
                };

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
                    .for_each(|numeric_update| {
                        self.renderer.update_valid(MetricState::Numeric(
                            numeric_update.entry,
                            numeric_update.numeric_entry,
                        ))
                    });

                if let Some(logger) = &mut self.progress_logger {
                    logger.update_split(item.progress.items_processed);
                    logger.log_event_training("Iteration".to_string());
                }
                self.renderer.update_split(item.progress.items_processed);
                self.renderer.log_event_training("Iteration".to_string());
            }
            LearnerEvent::EndSplit(epoch) => {
                self.store
                    .add_event_valid(crate::metric::store::Event::EndEpoch(EpochSummary::new(
                        epoch,
                        Split::Valid,
                    )));
                if let Some(logger) = &mut self.progress_logger {
                    logger.end_split();
                }
                self.renderer.end_split();
                self.metrics.end_epoch_valid();
            }
            LearnerEvent::EndEpoch(_) => {} // update_epoch is handled in process_train(EndEpoch)
            LearnerEvent::End(_) => {}      // no-op
        }
    }
    fn renderer(self) -> Box<dyn MetricsRenderer> {
        self.renderer
    }
}
