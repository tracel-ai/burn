use burn_core::data::dataloader::Progress;
use burn_optim::lr_scheduler::module_lr_scheduler::ModuleLearningRate;
use burn_std::ExecutionError;

use super::EventProcessorError;

use crate::{
    LearnerSummary,
    renderer::{EvaluationName, MetricsRenderer},
};

/// Event happening during the training/validation process.
pub enum LearnerEvent<T> {
    /// Signal the start of the process (e.g., training start).
    Start {
        /// The total number of training epochs.
        total_epochs: usize,
        /// The starting epoch.
        starting_epoch: usize,
        /// An optional label for this training.
        label: Option<String>,
    },
    /// Signal that an item have been processed.
    ProcessedItem(TrainingItem<T>),
    /// Signal the start of a split.
    StartSplit {
        /// The epoch number.
        epoch_number: usize,
        /// The total number of items to be processed during this split.
        total_items: usize,
    },
    /// Signal the end of a split, carrying the current epoch number.
    EndSplit(usize),
    /// Signal the end of a full epoch.
    EndEpoch(usize),
    /// Signal the end of the process (e.g., training end).
    End(Option<LearnerSummary>),
}

/// Event happening during the evaluation process.
pub enum EvaluatorEvent<T> {
    /// Signal the start of the process (e.g., evaluation start)
    Start {
        /// The total number of items to evaluate.
        total_tests: usize,
    },
    /// Signal the start of a test split, carrying the split name and total number of items.
    StartTest(EvaluationName, usize),
    /// Signal that an item have been processed.
    ProcessedItem(EvaluationName, EvaluationItem<T>),
    /// Signal the end of a single test split.
    EndTest,
    /// Signal the end of the process (e.g., evaluation end).
    End(Option<LearnerSummary>),
}

/// Items that are lazy are not ready to be processed by metrics.
///
/// We want to sync them on a different thread to avoid blocking training.
pub trait ItemLazy: Send + Sized {
    /// Sync the item.
    ///
    /// # Errors
    ///
    /// Returns an [`ExecutionError`] when the item's pending work cannot be dispatched, e.g. on a
    /// device that is poisoned.
    fn sync(self) -> Result<Self, ExecutionError>;
}

/// Process events happening during training and validation.
pub trait EventProcessorTraining<TrainEvent, ValidEvent>: Send {
    /// Collect a training event.
    ///
    /// # Errors
    ///
    /// Returns a [`ProcessorError`] listing every metric that could not process the event.
    /// The other metrics still processed it. Note that an asynchronous processor reports
    /// it on a later call instead (see [`AsyncProcessorTraining`](super::AsyncProcessorTraining)).
    fn process_train(&mut self, event: TrainEvent) -> Result<(), EventProcessorError>;
    /// Collect a validation event.
    ///
    /// # Errors
    ///
    /// Same as [`process_train`](Self::process_train).
    fn process_valid(&mut self, event: ValidEvent) -> Result<(), EventProcessorError>;
    /// Wait until previously submitted events are processed (no-op for sync processors).
    ///
    /// # Errors
    ///
    /// Returns a [`ProcessorError`] listing every metric failure among the events processed
    /// since the last error was reported.
    fn flush(&mut self) -> Result<(), EventProcessorError> {
        Ok(())
    }
    /// Returns the renderer used for training.
    fn renderer(self) -> Box<dyn MetricsRenderer>;
}

/// Process events happening during evaluation.
pub trait EventProcessorEvaluation: Send {
    /// The test item.
    type ItemTest: ItemLazy;

    /// Collect a test event.
    ///
    /// # Errors
    ///
    /// Returns a [`ProcessorError`] listing every metric that could not process the event.
    /// The other metrics still processed it. Note that an asynchronous processor reports
    /// it on a later call instead (see [`AsyncProcessorEvaluation`](super::AsyncProcessorEvaluation)).
    fn process_test(
        &mut self,
        event: EvaluatorEvent<Self::ItemTest>,
    ) -> Result<(), EventProcessorError>;

    /// Wait until previously submitted events are processed (no-op for sync processors).
    ///
    /// # Errors
    ///
    /// Returns a [`ProcessorError`] listing every metric failure among the events processed
    /// since the last error was reported.
    fn flush(&mut self) -> Result<(), EventProcessorError> {
        Ok(())
    }

    /// Returns the renderer used for evaluation.
    fn renderer(self) -> Box<dyn MetricsRenderer>;
}

/// A learner item.
#[derive(new)]
pub struct TrainingItem<T> {
    /// The item.
    pub item: T,

    /// The progress.
    pub progress: Progress,

    /// The iteration, if it it different from the items processed.
    pub iteration: Option<usize>,

    /// The learning rate for a module's parameters.
    pub lr: Option<ModuleLearningRate>,
}

impl<T: ItemLazy> ItemLazy for TrainingItem<T> {
    fn sync(self) -> Result<Self, ExecutionError> {
        Ok(TrainingItem {
            item: self.item.sync()?,
            progress: self.progress,
            iteration: self.iteration,
            lr: self.lr,
        })
    }
}

/// An evaluation item.
#[derive(new)]
pub struct EvaluationItem<T> {
    /// The item.
    pub item: T,

    /// The progress.
    pub progress: Progress,

    /// The iteration, if it it different from the items processed.
    pub iteration: Option<usize>,
}

impl<T: ItemLazy> ItemLazy for EvaluationItem<T> {
    fn sync(self) -> Result<Self, ExecutionError> {
        Ok(EvaluationItem {
            item: self.item.sync()?,
            progress: self.progress,
            iteration: self.iteration,
        })
    }
}

impl ItemLazy for () {
    fn sync(self) -> Result<Self, ExecutionError> {
        Ok(())
    }
}
