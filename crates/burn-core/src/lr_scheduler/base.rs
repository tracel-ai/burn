use burn_tensor::backend::Backend;

use crate::{record::Record, LearningRate};

/// Learning rate scheduler defines how the learning rate will evolve during training.
pub trait LrScheduler: Send + Sync {
    /// Scheduler associative type to be used when saving and loading the state.
    type Record<B: Backend>: Record<B>;

    /// Perform the scheduler step, potentially updating its state, and returning the effective
    /// learning rate.
    fn step(&mut self) -> LearningRate;

    /// Get the current state of the scheduler as a [record](Record).
    fn to_record<B: Backend>(&self) -> Self::Record<B>;

    /// Load the state of the scheduler as a [record](Record).
    fn load_record<B: Backend>(self, record: Self::Record<B>) -> Self;
}
