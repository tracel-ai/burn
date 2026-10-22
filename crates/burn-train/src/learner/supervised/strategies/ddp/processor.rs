use burn_core::data::dataloader::Progress;

use crate::{EventProcessorTraining, Interrupter, LearnerEvent, TrainingItem};

/// Aggregates reports for one epoch. Worker barriers keep epochs and splits separate,
/// so every submitted item goes directly to the underlying event processor.
pub(crate) struct WorkerEventProcessor<P> {
    processor: P,
    totals: Vec<Option<usize>>,
    snapshots: Vec<Progress>,
    iteration: usize,
    interrupter: Interrupter,
    /// Set by the main worker before releasing the epoch barrier, including early stopping.
    pub stop_after_epoch: bool,
}

impl<P> WorkerEventProcessor<P> {
    pub fn new(processor: P, totals: Vec<Option<usize>>, interrupter: Interrupter) -> Self {
        Self {
            processor,
            totals,
            snapshots: Vec::new(),
            iteration: 0,
            interrupter,
            stop_after_epoch: false,
        }
    }

    pub fn start_epoch<T, V>(&mut self, epoch: usize)
    where
        P: EventProcessorTraining<LearnerEvent<T>, LearnerEvent<V>>,
    {
        self.snapshots = self
            .totals
            .iter()
            .map(|total| Progress::new(0, *total, Some("items".into())))
            .collect();
        self.iteration = 0;
        self.interrupter
            .fail_on_error(self.processor.process_train(LearnerEvent::StartSplit {
                epoch_number: epoch,
                progress: Progress::aggregate(&self.snapshots),
            }));
    }

    pub fn train<T, V>(&mut self, worker: usize, mut item: TrainingItem<T>)
    where
        P: EventProcessorTraining<LearnerEvent<T>, LearnerEvent<V>>,
    {
        self.snapshots[worker] = item.progress;
        self.iteration += 1;
        item.progress = Progress::aggregate(&self.snapshots);
        item.iteration = Some(self.iteration);
        self.interrupter.fail_on_error(
            self.processor
                .process_train(LearnerEvent::ProcessedItem(item)),
        );
    }

    pub fn end_train(&mut self, worker: usize, progress: Progress) {
        self.snapshots[worker] = progress;
    }

    /// Called by the main worker after every worker has published its final snapshot.
    pub fn finish_train<T, V>(&mut self, epoch: usize)
    where
        P: EventProcessorTraining<LearnerEvent<T>, LearnerEvent<V>>,
    {
        self.flush();

        let mut progress = Progress::aggregate(&self.snapshots);
        if !self.interrupter.should_stop() {
            progress.resolve_total();
        }
        self.interrupter.fail_on_error(
            self.processor
                .process_train(LearnerEvent::EndSplit { epoch, progress }),
        );
        self.flush();
    }

    pub fn valid<T, V>(&mut self, event: LearnerEvent<V>)
    where
        P: EventProcessorTraining<LearnerEvent<T>, LearnerEvent<V>>,
    {
        self.interrupter
            .fail_on_error(self.processor.process_valid(event));
    }

    pub fn end_epoch<T, V>(&mut self, epoch: usize)
    where
        P: EventProcessorTraining<LearnerEvent<T>, LearnerEvent<V>>,
    {
        self.flush();
        if !self.interrupter.should_stop() {
            self.interrupter
                .fail_on_error(self.processor.process_train(LearnerEvent::EndEpoch(epoch)));
            self.flush();
        }
    }

    pub fn flush<T, V>(&mut self)
    where
        P: EventProcessorTraining<LearnerEvent<T>, LearnerEvent<V>>,
    {
        self.interrupter.fail_on_error(self.processor.flush());
    }

    pub fn into_inner(self) -> P {
        self.processor
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EventProcessorError, metric::store::Split, renderer::MetricsRenderer};

    #[derive(Default)]
    struct RecordingProcessor {
        events: Vec<LearnerEvent<()>>,
        fail_flush: bool,
    }

    impl EventProcessorTraining<LearnerEvent<()>, LearnerEvent<()>> for RecordingProcessor {
        fn process_train(&mut self, event: LearnerEvent<()>) -> Result<(), EventProcessorError> {
            self.events.push(event);
            Ok(())
        }

        fn flush(&mut self) -> Result<(), EventProcessorError> {
            if self.fail_flush {
                Err(EventProcessorError::sync(
                    Split::Train,
                    burn_std::ExecutionError::with_context("metric failed"),
                ))
            } else {
                Ok(())
            }
        }

        fn process_valid(&mut self, _: LearnerEvent<()>) -> Result<(), EventProcessorError> {
            Ok(())
        }

        fn renderer(self) -> Box<dyn MetricsRenderer> {
            unreachable!()
        }
    }

    fn item(count: usize) -> TrainingItem<()> {
        TrainingItem::new((), Progress::new(count, None, None), Some(1), None)
    }

    #[test]
    fn aggregates_final_worker_counts_and_resets_at_the_next_epoch() {
        let processor = RecordingProcessor::default();
        let mut reporting = WorkerEventProcessor::new(processor, vec![None; 2], Interrupter::new());
        reporting.start_epoch(3);
        reporting.train(1, item(3));
        reporting.end_train(1, Progress::new(3, 3, None));
        reporting.train(0, item(2));
        reporting.end_train(0, Progress::new(2, 2, None));
        reporting.finish_train(3);
        reporting.end_epoch(3);
        // An empty epoch must neither retain prior counts nor fabricate an iteration.
        reporting.start_epoch(4);
        reporting.end_train(0, Progress::new(0, 0, None));
        reporting.end_train(1, Progress::new(0, 0, None));
        reporting.finish_train(4);
        reporting.end_epoch(4);

        let events = reporting.into_inner().events;
        assert!(matches!(events.as_slice(), [
            LearnerEvent::StartSplit { epoch_number: 3, .. },
            LearnerEvent::ProcessedItem(first),
            LearnerEvent::ProcessedItem(second),
            LearnerEvent::EndSplit { epoch: 3, progress: final_progress },
            LearnerEvent::EndEpoch(3),
            LearnerEvent::StartSplit { epoch_number: 4, progress },
            LearnerEvent::EndSplit { epoch: 4, progress: empty },
            LearnerEvent::EndEpoch(4),
        ] if first.iteration == Some(1) && first.progress.items_processed == 3
            && second.iteration == Some(2) && second.progress.items_processed == 5
            && final_progress.total == Some(5)
            && progress.items_processed == 0 && empty.total == Some(0)
            && empty.items_processed == 0));
    }

    #[test]
    fn flush_observes_worker_errors_before_the_final_update_decision() {
        let processor = RecordingProcessor {
            fail_flush: true,
            ..Default::default()
        };
        let interrupter = Interrupter::new();
        let mut reporting =
            WorkerEventProcessor::new(processor, vec![None; 2], interrupter.clone());
        reporting.start_epoch(1);
        reporting.train(1, item(1));
        assert!(!interrupter.should_stop());
        reporting.flush();
        assert!(interrupter.should_stop());
        reporting.finish_train(1);
        reporting.end_epoch(1);
        let events = reporting.into_inner().events;
        assert!(
            matches!(events.last(), Some(LearnerEvent::EndSplit { progress, .. })
            if progress.total == None)
        );
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, LearnerEvent::EndEpoch(_)))
        );
    }
}
