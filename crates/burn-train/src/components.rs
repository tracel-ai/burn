use crate::{EventProcessorTraining, checkpoint::CheckpointingStrategy, metric::ItemLazy};
use burn_core::{
    data::dataloader::DataLoader, module::AutodiffModule, tensor::backend::AutodiffBackend,
};
use burn_optim::{Optimizer, lr_scheduler::LrScheduler};
use std::{marker::PhantomData, sync::Arc};

/// Components used for a model to learn, grouped in one trait.
pub trait LearningComponentsTypes: Clone {
    /// The backend used for learning.
    type Backend: AutodiffBackend;
    /// The learning rate scheduler used for learning.
    type LrScheduler: LrScheduler + 'static;
    /// The model that learns.
    type Model: AutodiffModule<Self::Backend, InnerModule = Self::InnerModel>
        + core::fmt::Display
        + 'static;
    /// The non-autodiff type of the model.
    type InnerModel;
    /// The optimizer used for learning.
    type Optimizer: Optimizer<Self::Model, Self::Backend> + 'static;
}

#[derive(Clone)]
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
    M: AutodiffModule<B> + core::fmt::Display + 'static,
    O: Optimizer<M, B> + 'static,
{
    type Backend = B;
    type LrScheduler = LR;
    type Model = M;
    type InnerModel = M::InnerModule;
    type Optimizer = O;
}

/// The training backend.
pub type TrainBackend<LC> = <LC as LearningComponentsTypes>::Backend;
/// The validation backend.
pub type ValidBackend<LC> =
    <<LC as LearningComponentsTypes>::Backend as AutodiffBackend>::InnerBackend;
/// Type for training input
pub(crate) type InputTrain<LD> = <LD as LearningData>::TrainInput;
/// Type for validation input
pub(crate) type InputValid<LD> = <LD as LearningData>::ValidInput;
/// Type for training output
pub(crate) type OutputTrain<LD> = <LD as LearningData>::TrainOutput;
/// Type for validation output
pub(crate) type OutputValid<LD> = <LD as LearningData>::ValidOutput;
/// A reference to the training split [DataLoader](DataLoader).
pub type TrainLoader<LC, LD> = Arc<dyn DataLoader<TrainBackend<LC>, InputTrain<LD>>>;
/// A reference to the validation split [DataLoader](DataLoader).
pub type ValidLoader<LC, LD> = Arc<dyn DataLoader<ValidBackend<LC>, InputValid<LD>>>;

/// Regroups types of input and outputs for training and validation
pub trait LearningData {
    /// Type of input to the training stop
    type TrainInput: Send + 'static;
    /// Type of input to the validation step
    type ValidInput: Send + 'static;
    /// Type of output of the training step
    type TrainOutput: ItemLazy + 'static;
    /// Type of output of the validation step
    type ValidOutput: ItemLazy + 'static;
}

/// Concrete type that implements [LearningData](LearningData) trait.
pub struct LearningDataMarker<TI, VI, TO, VO> {
    _phantom_data: PhantomData<(TI, VI, TO, VO)>,
}

impl<TI, VI, TO, VO> LearningData for LearningDataMarker<TI, VI, TO, VO>
where
    TI: Send + 'static,
    VI: Send + 'static,
    TO: ItemLazy + 'static,
    VO: ItemLazy + 'static,
{
    type TrainInput = TI;
    type ValidInput = VI;
    type TrainOutput = TO;
    type ValidOutput = VO;
}

/// All components used to execute a learning paradigm, grouped in one trait.
pub trait ParadigmComponentsTypes {
    /// The data processed by the learning model during training and validation.
    type LearningData: LearningData;
    /// Processes events happening during training and validation.
    type EventProcessor: EventProcessorTraining<
            ItemTrain = <Self::LearningData as LearningData>::TrainOutput,
            ItemValid = <Self::LearningData as LearningData>::ValidOutput,
        > + 'static;
    /// The strategy used to save and delete checkpoints.
    type CheckpointerStrategy: CheckpointingStrategy;
}

/// Concrete type that implements the [ParadigmComponentsTypes](ParadigmComponentsTypes) trait.
pub struct ParadigmComponentsMarker<LD, EP, CS> {
    _learning_data: PhantomData<LD>,
    _event_processor: PhantomData<EP>,
    _strategy: PhantomData<CS>,
}

impl<LD, EP, CS> ParadigmComponentsTypes for ParadigmComponentsMarker<LD, EP, CS>
where
    LD: LearningData,
    EP: EventProcessorTraining<
            ItemTrain = <LD as LearningData>::TrainOutput,
            ItemValid = <LD as LearningData>::ValidOutput,
        > + 'static,
    CS: CheckpointingStrategy,
{
    type LearningData = LD;
    type EventProcessor = EP;
    type CheckpointerStrategy = CS;
}
