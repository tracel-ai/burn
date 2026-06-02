use burn_core::data::dataloader::Progress;
use burn_optim::LearningRate;
use core::any::Any;

use crate::{
    LearnerSummary,
    renderer::{EvaluationName, MetricsRenderer},
};

/// Event happening during the training/validation process.
///
/// The event is intentionally **not** generic over the model's output type: the
/// concrete output is erased into a [`Box<dyn ErasedItem>`](ErasedItem) the
/// moment it enters the pipeline (see [`TrainingItem::new`]). This keeps the
/// async worker, processors, metric collections, store and renderer compiled
/// exactly once, no matter how many model/output types a binary trains.
pub enum LearnerEvent {
    /// Signal the start of the process (e.g., training start).
    Start {
        /// The total number of training epochs.
        total_epochs: usize,
    },
    /// Signal that an item have been processed.
    ProcessedItem(TrainingItem),
    /// Signal the start of a split, carrying the total number of items in that split.
    StartSplit(usize),
    /// Signal the end of a split, carrying the current epoch number.
    EndSplit(usize),
    /// Signal the end of a full epoch.
    EndEpoch(usize),
    /// Signal the end of the process (e.g., training end).
    End(Option<LearnerSummary>),
}

/// Event happening during the evaluation process.
pub enum EvaluatorEvent {
    /// Signal the start of the process (e.g., evaluation start)
    Start {
        /// The total number of items to evaluate.
        total_tests: usize,
    },
    /// Signal the start of a test split, carrying the split name and total number of items.
    StartTest(EvaluationName, usize),
    /// Signal that an item have been processed.
    ProcessedItem(EvaluationName, EvaluationItem),
    /// Signal the end of a single test split.
    EndTest,
    /// Signal the end of the process (e.g., evaluation end).
    End(Option<LearnerSummary>),
}

/// Items that are lazy are not ready to be processed by metrics.
///
/// We want to sync them on a different thread to avoid blocking training.
pub trait ItemLazy: Send {
    /// Sync the item.
    fn sync(self) -> Self;
}

/// Object-safe, type-erased view over an [`ItemLazy`] value.
///
/// The event/metric pipeline holds the model output as a `dyn ErasedItem` so
/// that none of the pipeline machinery is monomorphized over the concrete
/// output type. The only code still specialized per output type is the blanket
/// `sync` shim below and the downcast-and-adapt step in `MetricWrapper`.
pub(crate) trait ErasedItem: Send {
    /// Sync the underlying lazy item (see [`ItemLazy::sync`]), preserving erasure.
    fn sync_erased(self: Box<Self>) -> Box<dyn ErasedItem>;
    /// Borrow the underlying item so metric updaters can downcast to it.
    fn as_any(&self) -> &dyn Any;
}

impl<T: ItemLazy + 'static> ErasedItem for T {
    fn sync_erased(self: Box<Self>) -> Box<dyn ErasedItem> {
        Box::new((*self).sync())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Process events happening during training and validation.
pub trait EventProcessorTraining<TrainEvent, ValidEvent>: Send {
    /// Collect a training event.
    fn process_train(&mut self, event: TrainEvent);
    /// Collect a validation event.
    fn process_valid(&mut self, event: ValidEvent);
    /// Returns the renderer used for training.
    fn renderer(self) -> Box<dyn MetricsRenderer>;
}

/// Process events happening during evaluation.
pub trait EventProcessorEvaluation: Send {
    /// Collect a test event.
    fn process_test(&mut self, event: EvaluatorEvent);

    /// Returns the renderer used for evaluation.
    fn renderer(self) -> Box<dyn MetricsRenderer>;
}

/// A learner item.
///
/// The model output is type-erased on construction (see [`TrainingItem::new`])
/// so this type — and the whole pipeline downstream of it — is not generic over
/// the output type.
pub struct TrainingItem {
    /// The type-erased item.
    pub(crate) item: Box<dyn ErasedItem>,

    /// The progress.
    pub progress: Progress,

    /// The iteration, if it it different from the items processed.
    pub iteration: Option<usize>,

    /// The learning rate.
    pub lr: Option<LearningRate>,
}

impl TrainingItem {
    /// Create a new training item, erasing the concrete model output type.
    pub fn new<T: ItemLazy + 'static>(
        item: T,
        progress: Progress,
        iteration: Option<usize>,
        lr: Option<LearningRate>,
    ) -> Self {
        Self {
            item: Box::new(item),
            progress,
            iteration,
            lr,
        }
    }

    /// Sync the underlying lazy item.
    pub(crate) fn sync(mut self) -> Self {
        self.item = self.item.sync_erased();
        self
    }
}

/// An evaluation item.
pub struct EvaluationItem {
    /// The type-erased item.
    pub(crate) item: Box<dyn ErasedItem>,

    /// The progress.
    pub progress: Progress,

    /// The iteration, if it it different from the items processed.
    pub iteration: Option<usize>,
}

impl EvaluationItem {
    /// Create a new evaluation item, erasing the concrete model output type.
    pub fn new<T: ItemLazy + 'static>(
        item: T,
        progress: Progress,
        iteration: Option<usize>,
    ) -> Self {
        Self {
            item: Box::new(item),
            progress,
            iteration,
        }
    }

    /// Sync the underlying lazy item.
    pub(crate) fn sync(mut self) -> Self {
        self.item = self.item.sync_erased();
        self
    }
}

impl ItemLazy for () {
    fn sync(self) -> Self {}
}
