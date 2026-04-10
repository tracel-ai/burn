use crate::{InferenceStep, TrainStep};
use burn_core::module::AutodiffModule;
use burn_optim::{Optimizer, lr_scheduler::LrScheduler};
use std::marker::PhantomData;

/// Components used for a model to learn, grouped in one trait.
pub trait LearningComponentsTypes {
    /// The learning rate scheduler used for training.
    type LrScheduler: LrScheduler + 'static;
    /// The model to train.
    type Model: TrainStep + InferenceStep + AutodiffModule + core::fmt::Display + 'static;
    /// The optimizer used for training.
    type Optimizer: Optimizer<Self::Model> + 'static;
}

/// Concrete type that implements the [LearningComponentsTypes](LearningComponentsTypes) trait.
pub struct LearningComponentsMarker<LR, M, O> {
    _lr_scheduler: PhantomData<LR>,
    _model: PhantomData<M>,
    _optimizer: PhantomData<O>,
}

impl<LR, M, O> LearningComponentsTypes for LearningComponentsMarker<LR, M, O>
where
    LR: LrScheduler + 'static,
    M: TrainStep + InferenceStep + AutodiffModule + core::fmt::Display + 'static,
    O: Optimizer<M> + 'static,
{
    type LrScheduler = LR;
    type Model = M;
    type Optimizer = O;
}

/// The model used for training.
pub type TrainingModel<LC> = <LC as LearningComponentsTypes>::Model;
/// The non-autodiff model.
pub(crate) type InferenceModel<LC> = <LC as LearningComponentsTypes>::Model;
/// Type for training input.
pub(crate) type TrainingModelInput<LC> =
    <<LC as LearningComponentsTypes>::Model as TrainStep>::Input;
/// Type for inference input.
pub(crate) type InferenceModelInput<LC> =
    <<LC as LearningComponentsTypes>::Model as InferenceStep>::Input;
/// Type for training output.
pub(crate) type TrainingModelOutput<LC> =
    <<LC as LearningComponentsTypes>::Model as TrainStep>::Output;
/// Type for inference output.
pub(crate) type InferenceModelOutput<LC> =
    <<LC as LearningComponentsTypes>::Model as InferenceStep>::Output;
