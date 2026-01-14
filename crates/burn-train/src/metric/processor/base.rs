use burn_core::data::dataloader::Progress;
use burn_optim::LearningRate;

use crate::{
    EpisodeSummary, LearnerSummary,
    renderer::{EvaluationName, MetricsRenderer},
};

/// Event happening during the training/validation process.
pub enum LearnerEvent<T> {
    /// Signal the start of the process (e.g., training start)
    Start,
    /// Signal that an item have been processed.
    ProcessedItem(LearnerItem<T>),
    /// Signal the end of an epoch.
    EndEpoch(usize),
    /// Signal the end of the process (e.g., training end).
    End(Option<LearnerSummary>),
}

/// Event happening during reinforcement learning.
pub enum RLEvent<TS, ES> {
    /// Signal the start of the process (e.g., learning starts).
    Start,
    /// Signal an agent's training step.
    TrainStep(LearnerItem<TS>),
    /// Signal a timestep of the agent-environement interface.
    TimeStep(LearnerItem<ES>),
    /// Signal an episode end.
    EpisodeEnd(LearnerItem<EpisodeSummary>),
    /// Signal the end of the process (e.g., learning ends).
    End(Option<LearnerSummary>),
}

/// Event happening during evaluation of a renforcement learning's agent.
pub enum AgentEvaluationEvent<T> {
    /// Signal the start of the process (e.g., training start)
    Start,
    /// Signal a timestep of the agent-environement interface.
    TimeStep(LearnerItem<T>),
    /// Signal an episode end.
    EpisodeEnd(LearnerItem<EpisodeSummary>),
    /// Signal the end of the process (e.g., training end).
    End,
}

/// Event happening during the evaluation process.
pub enum EvaluatorEvent<T> {
    /// Signal the start of the process (e.g., training start)
    Start,
    /// Signal that an item have been processed.
    ProcessedItem(EvaluationName, LearnerItem<T>),
    /// Signal the end of the process (e.g., training end).
    End,
}

/// Items that are lazy are not ready to be processed by metrics.
///
/// We want to sync them on a different thread to avoid blocking training.
pub trait ItemLazy: Send {
    /// Item that is properly synced and ready to be processed by metrics.
    type ItemSync: Send;

    /// Sync the item.
    fn sync(self) -> Self::ItemSync;
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
    /// The test item.
    type ItemTest: ItemLazy;

    /// Collect a test event.
    fn process_test(&mut self, event: EvaluatorEvent<Self::ItemTest>);

    /// Returns the renderer used for evaluation.
    fn renderer(self) -> Box<dyn MetricsRenderer>;
}

/// A learner item.
#[derive(new)]
pub struct LearnerItem<T> {
    /// The item.
    pub item: T,

    /// The progress.
    pub progress: Progress,

    /// The epoch.
    pub epoch: usize,

    /// The total number of epochs.
    pub epoch_total: usize,

    /// The iteration.
    pub iteration: usize,

    /// The learning rate.
    pub lr: Option<LearningRate>,
}

impl<T: ItemLazy> ItemLazy for LearnerItem<T> {
    type ItemSync = LearnerItem<T::ItemSync>;

    fn sync(self) -> Self::ItemSync {
        LearnerItem {
            item: self.item.sync(),
            progress: self.progress,
            epoch: self.epoch,
            epoch_total: self.epoch_total,
            iteration: self.iteration,
            lr: self.lr,
        }
    }
}

impl ItemLazy for () {
    type ItemSync = ();

    fn sync(self) -> Self::ItemSync {}
}
