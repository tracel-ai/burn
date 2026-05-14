use crate::renderer::TrainingProgress;

/// Trait for logging training progress at each step and end of epoch.
pub trait TrainingProgressLogger: Send {
    /// Log the progress of the current training step.
    fn update_train(&mut self, progress: &TrainingProgress);

    /// Log the progress of the current validation step.
    fn update_valid(&mut self, progress: &TrainingProgress);

    /// Called at the end of an epoch evaluation.
    fn end_evaluation(&mut self, epoch: usize);
}
