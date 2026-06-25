use super::{EventProcessorTraining, ItemLazy, LearnerEvent, MetricsTraining};
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
                /*
                In compute:
                NumericMetricUpdate {
                    entry: update,
                    // Current value is not applicable. This is the final epoch-level value computed.
                    numeric_entry: None,
                    running_entry: running, /* NumericEntry */
                }

                TUI: NumericEntry
                */
                let state = match numeric_update.numeric_entry {
                    Some(value) => MetricState::Numeric(numeric_update.entry, value),
                    None => match numeric_update.running_entry {
                        Some(value) => MetricState::Numeric(numeric_update.entry, value),
                        None => MetricState::Generic(numeric_update.entry),
                    },
                };
                self.renderer.update_train(state)
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
                let state = match numeric_update.numeric_entry {
                    Some(value) => MetricState::Numeric(numeric_update.entry, value),
                    None => match numeric_update.running_entry {
                        Some(value) => MetricState::Numeric(numeric_update.entry, value),
                        None => MetricState::Generic(numeric_update.entry),
                    },
                };
                self.renderer.update_valid(state)
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
                        let state = match numeric_update.numeric_entry {
                            Some(value) => MetricState::Numeric(numeric_update.entry, value),
                            None => MetricState::Generic(numeric_update.entry),
                        };
                        self.renderer.update_test(name.clone(), state)
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
            LearnerEvent::Start {
                total_epochs,
                starting_epoch,
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
                    logger.start(total_epochs, starting_epoch, None);
                }
                self.renderer.start(total_epochs, starting_epoch, None);
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
                let item = item.sync();
                let metadata = MetricMetadata {
                    progress: item.progress.clone(),
                    iteration: item.iteration,
                    lr: item.lr.clone(),
                };

                let update = self.metrics.update_train(&item, &metadata);
                self.handle_train_metrics_update(update);

                if let Some(logger) = &mut self.progress_logger {
                    logger.update_split(item.progress.items_processed);
                    logger.log_event_training("Iteration".to_string());
                }
                self.renderer.update_split(item.progress.items_processed);
                self.renderer.log_event_training("Iteration".to_string());
            }
            LearnerEvent::EndSplit(epoch) => {
                let update = self.metrics.end_epoch_train();
                self.handle_train_metrics_update(update);

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
    }

    fn process_valid(&mut self, event: LearnerEvent<V>) {
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
                let item = item.sync();
                let metadata = MetricMetadata {
                    progress: item.progress.clone(),
                    iteration: item.iteration,
                    lr: item.lr.clone(),
                };

                let update = self.metrics.update_valid(&item, &metadata);
                self.handle_valid_metrics_update(update);

                if let Some(logger) = &mut self.progress_logger {
                    logger.update_split(item.progress.items_processed);
                    logger.log_event_training("Iteration".to_string());
                }
                self.renderer.update_split(item.progress.items_processed);
                self.renderer.log_event_training("Iteration".to_string());
            }
            LearnerEvent::EndSplit(epoch) => {
                let update = self.metrics.end_epoch_valid();
                self.handle_valid_metrics_update(update);

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
    }
    fn renderer(self) -> Box<dyn MetricsRenderer> {
        self.renderer
    }
}
