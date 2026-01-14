use crate::{EpisodeSummary, ItemLazy, LearnerItem, LearnerSummary, renderer::MetricsRenderer};

pub enum RlTrainingEvent<TS, ES> {
    /// Signal the start of the process (e.g., training start)
    Start,
    /// Signal that an item have been processed.
    TrainStep(LearnerItem<TS>),
    EnvStep(LearnerItem<ES>),
    EpisodeEnd(LearnerItem<EpisodeSummary>),
    /// Signal the end of the process (e.g., training end).
    End(Option<LearnerSummary>),
}

/// Event happening during the evaluation process.
pub enum RlEvaluationEvent<T> {
    /// Signal the start of the process (e.g., training start)
    Start,
    EnvStep(LearnerItem<T>),
    EpisodeEnd(LearnerItem<EpisodeSummary>),
    /// Signal the end of the process (e.g., training end).
    End,
}

/// Process events happening during training and validation.
pub trait RlEventProcessorTrain: Send {
    /// The training item.
    type TrainingOutput: ItemLazy;
    /// The validation item.
    type ActionContext: ItemLazy;

    /// Collect a training event.
    fn process_train(&mut self, event: RlTrainingEvent<Self::TrainingOutput, Self::ActionContext>);
    /// Collect a validation event.
    fn process_valid(&mut self, event: RlEvaluationEvent<Self::ActionContext>);
    /// Returns the renderer used for training.
    fn renderer(self) -> Box<dyn MetricsRenderer>;
}
