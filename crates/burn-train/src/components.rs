use crate::{InferenceStep, TrainStep};
use burn_core::{module::AutodiffModule, tensor::backend::AutodiffBackend};
use burn_optim::{Optimizer, lr_scheduler::LrScheduler};
use std::marker::PhantomData;

/// Components used for a model to learn, grouped in one trait.
pub trait LearningComponentsTypes {
    /// The backend used for training.
    type Backend: AutodiffBackend;
    /// The learning rate scheduler used for training.
    type LrScheduler: LrScheduler + 'static;
    /// The model to train.
    type TrainingModel: TrainStep
        + AutodiffModule<Self::Backend, InnerModule = Self::InferenceModel>
        + core::fmt::Display
        + 'static;
    /// The non-autodiff type of the model.
    type InferenceModel: InferenceStep;
    /// The optimizer used for training.
    type Optimizer: Optimizer<Self::TrainingModel, Self::Backend> + 'static;
}

/// Concrete type that implements the [LearningComponentsTypes](LearningComponentsTypes) trait.
pub struct LearningComponentsMarker<B, LR, M, O> {
    _backend: PhantomData<B>,
    _lr_scheduler: PhantomData<LR>,
    _model: PhantomData<M>,
    _optimizer: PhantomData<O>,
}

impl<B, LR, M, O> LearningComponentsTypes for LearningComponentsMarker<B, LR, M, O>
where
    B: AutodiffBackend,
    LR: LrScheduler + 'static,
    M: TrainStep + AutodiffModule<B> + core::fmt::Display + 'static,
    M::InnerModule: InferenceStep,
    O: Optimizer<M, B> + 'static,
{
    type Backend = B;
    type LrScheduler = LR;
    type TrainingModel = M;
    type InferenceModel = M::InnerModule;
    type Optimizer = O;
}

/// The training backend.
pub type TrainingBackend<LC> = <LC as LearningComponentsTypes>::Backend;
/// The inference backend.
pub(crate) type InferenceBackend<LC> =
    <<LC as LearningComponentsTypes>::Backend as AutodiffBackend>::InnerBackend;
/// The model used for training.
pub type TrainingModel<LC> = <LC as LearningComponentsTypes>::TrainingModel;
/// The non-autodiff model.
pub(crate) type InferenceModel<LC> = <LC as LearningComponentsTypes>::InferenceModel;
/// Type for training input.
pub(crate) type TrainingModelInput<LC> =
    <<LC as LearningComponentsTypes>::TrainingModel as TrainStep>::Input;
/// Type for inference input.
pub(crate) type InferenceModelInput<LC> =
    <<LC as LearningComponentsTypes>::InferenceModel as InferenceStep>::Input;
/// Type for training output.
pub(crate) type TrainingModelOutput<LC> =
    <<LC as LearningComponentsTypes>::TrainingModel as TrainStep>::Output;
/// Type for inference output.
pub(crate) type InferenceModelOutput<LC> =
    <<LC as LearningComponentsTypes>::InferenceModel as InferenceStep>::Output;
