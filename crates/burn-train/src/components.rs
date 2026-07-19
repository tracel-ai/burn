use crate::{InferenceStep, TrainStep};
use burn_core::module::AutodiffModule;

/// A single keyword trait for a Burn [module](burn_core::module::Module) used for learning.
pub trait LearnerModel:
    TrainStep + InferenceStep + AutodiffModule + core::fmt::Display + 'static
{
}

impl<T> LearnerModel for T where
    T: TrainStep + InferenceStep + AutodiffModule + core::fmt::Display + 'static
{
}

/// Type for training input.
pub(crate) type TrainingModelInput<M> = <M as TrainStep>::Input;
/// Type for inference input.
pub(crate) type InferenceModelInput<M> = <M as InferenceStep>::Input;
/// Type for training output.
pub(crate) type TrainingModelOutput<M> = <M as TrainStep>::Output;
/// Type for inference output.
pub(crate) type InferenceModelOutput<M> = <M as InferenceStep>::Output;
