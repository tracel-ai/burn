use crate::{TrainStep, ValidStep, metric::ItemLazy};
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
    type TrainingModel: TrainStep<TrainingModelInput<Self>, TrainingModelOutput<Self>>
        + AutodiffModule<Self::Backend, InnerModule = Self::InferenceModel>
        + core::fmt::Display
        + 'static;
    /// The non-autodiff type of the model.
    type InferenceModel: ValidStep<InferenceModelInput<Self>, InferenceModelOutput<Self>>;
    /// The optimizer used for training.
    type Optimizer: Optimizer<Self::TrainingModel, Self::Backend> + 'static;
    /// The [ModelDataTypes](crate::ModelDataTypes) types.
    type ModelIO: ModelDataTypes;
}

/// Concrete type that implements the [LearningComponentsTypes](LearningComponentsTypes) trait.
pub struct LearningComponentsMarker<B, LR, M, O, LD> {
    _backend: PhantomData<B>,
    _lr_scheduler: PhantomData<LR>,
    _model: PhantomData<M>,
    _optimizer: PhantomData<O>,
    _learning_data: PhantomData<LD>,
}

impl<B, LR, M, O, MD> LearningComponentsTypes for LearningComponentsMarker<B, LR, M, O, MD>
where
    B: AutodiffBackend,
    LR: LrScheduler + 'static,
    M: TrainStep<MD::TrainInput, MD::TrainOutput>
        + AutodiffModule<B>
        + core::fmt::Display
        + 'static,
    M::InnerModule: ValidStep<MD::InferenceInput, MD::InferenceOutput>,
    O: Optimizer<M, B> + 'static,
    MD: ModelDataTypes,
{
    type Backend = B;
    type LrScheduler = LR;
    type TrainingModel = M;
    type InferenceModel = M::InnerModule;
    type Optimizer = O;
    type ModelIO = MD;
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
    <<LC as LearningComponentsTypes>::ModelIO as ModelDataTypes>::TrainInput;
/// Type for inference input.
pub(crate) type InferenceModelInput<LC> =
    <<LC as LearningComponentsTypes>::ModelIO as ModelDataTypes>::InferenceInput;
/// Type for training output.
pub(crate) type TrainingModelOutput<LC> =
    <<LC as LearningComponentsTypes>::ModelIO as ModelDataTypes>::TrainOutput;
/// Type for inference output.
pub(crate) type InferenceModelOutput<LC> =
    <<LC as LearningComponentsTypes>::ModelIO as ModelDataTypes>::InferenceOutput;

/// Regroups types of input and outputs for training and inference.
pub trait ModelDataTypes {
    /// Type of input for a step of the training stage.
    type TrainInput: Send + 'static;
    /// Type of input for an inference step.
    type InferenceInput: Send + 'static;
    /// Type of output for a step of the training stage.
    type TrainOutput: ItemLazy + 'static;
    /// Type of output for an inference step.
    type InferenceOutput: ItemLazy + 'static;
}

/// Concrete type that implements [LearningData](LearningData) trait.
pub struct ModelDataMarker<TI, VI, TO, VO> {
    _phantom_data: PhantomData<(TI, VI, TO, VO)>,
}

impl<TI, II, TO, IO> ModelDataTypes for ModelDataMarker<TI, II, TO, IO>
where
    TI: Send + 'static,
    II: Send + 'static,
    TO: ItemLazy + 'static,
    IO: ItemLazy + 'static,
{
    type TrainInput = TI;
    type InferenceInput = II;
    type TrainOutput = TO;
    type InferenceOutput = IO;
}
