use crate::{
    TrainStep, ValidStep,
    checkpoint::{Checkpointer, CheckpointingStrategy},
    metric::{ItemLazy, processor::EventProcessorTraining},
};
use burn_core::{
    module::{AutodiffModule, Module},
    tensor::backend::AutodiffBackend,
};
use burn_optim::{Optimizer, lr_scheduler::LrScheduler};
use std::marker::PhantomData;

/// All components necessary to train a model grouped in one trait.
pub trait LearnerComponentTypes {
    /// The backend in used for the training.
    type Backend: AutodiffBackend;
    /// The learning rate scheduler used for the training.
    type LrScheduler: LrScheduler;
    /// The model to train.
    type Model: AutodiffModule<Self::Backend, InnerModule = Self::InnerModel> + 'static;
    /// The non-autodiff type of the model, should implement ValidationStep
    type InnerModel;
    /// The optimizer used for the training.
    type Optimizer: Optimizer<Self::Model, Self::Backend>;
    /// The checkpointer used for the model.
    type CheckpointerModel: Checkpointer<<Self::Model as Module<Self::Backend>>::Record, Self::Backend>;
    /// The checkpointer used for the optimizer.
    type CheckpointerOptimizer: Checkpointer<
            <Self::Optimizer as Optimizer<Self::Model, Self::Backend>>::Record,
            Self::Backend,
        > + Send;
    /// The checkpointer used for the scheduler.
    type CheckpointerLrScheduler: Checkpointer<<Self::LrScheduler as LrScheduler>::Record<Self::Backend>, Self::Backend>;
    /// The object that processes events happening during training and valid.
    type EventProcessor: EventProcessorTraining<
            ItemTrain = <Self::LearningData as LearningData>::TrainOutput,
            ItemValid = <Self::LearningData as LearningData>::ValidOutput,
        > + 'static;
    /// The strategy to save and delete checkpoints.
    type CheckpointerStrategy: CheckpointingStrategy;
    /// The data used to perform training and validation.
    type LearningData: LearningData;
}

/// Concrete type that implements [training components trait](TrainingComponents).
pub struct LearnerComponentsMarker<B, LR, M, O, CM, CO, CS, EP, S, LD> {
    _backend: PhantomData<B>,
    _lr_scheduler: PhantomData<LR>,
    _model: PhantomData<M>,
    _optimizer: PhantomData<O>,
    _checkpointer_model: PhantomData<CM>,
    _checkpointer_optim: PhantomData<CO>,
    _checkpointer_scheduler: PhantomData<CS>,
    _event_processor: PhantomData<EP>,
    _strategy: S,
    _learning_data: PhantomData<LD>,
}

impl<B, LR, M, O, CM, CO, CS, EP, S, LD> LearnerComponentTypes
    for LearnerComponentsMarker<B, LR, M, O, CM, CO, CS, EP, S, LD>
where
    B: AutodiffBackend,
    LR: LrScheduler,
    M: AutodiffModule<B>
        + TrainStep<LD::TrainInput, LD::TrainOutput>
        + core::fmt::Display
        + 'static,
    M::InnerModule: ValidStep<LD::ValidInput, LD::ValidOutput>,
    O: Optimizer<M, B>,
    CM: Checkpointer<M::Record, B>,
    CO: Checkpointer<O::Record, B>,
    CS: Checkpointer<LR::Record<B>, B>,
    EP: EventProcessorTraining<ItemTrain = LD::TrainOutput, ItemValid = LD::ValidOutput> + 'static,
    S: CheckpointingStrategy,
    LD: LearningData,
{
    type Backend = B;
    type LrScheduler = LR;
    type Model = M;
    type InnerModel = M::InnerModule;
    type Optimizer = O;
    type CheckpointerModel = CM;
    type CheckpointerOptimizer = CO;
    type CheckpointerLrScheduler = CS;
    type EventProcessor = EP;
    type CheckpointerStrategy = S;
    type LearningData = LD;
}

/// The training backend.
pub type TrainBackend<LC> = <LC as LearnerComponentTypes>::Backend;

/// The validation backend.
pub type ValidBackend<LC> =
    <<LC as LearnerComponentTypes>::Backend as AutodiffBackend>::InnerBackend;

/// Type for training input
pub(crate) type InputTrain<LC> =
    <<LC as LearnerComponentTypes>::LearningData as LearningData>::TrainInput;

/// Type for validation input
pub(crate) type InputValid<LC> =
    <<LC as LearnerComponentTypes>::LearningData as LearningData>::ValidInput;

/// Type for training output
pub(crate) type OutputTrain<LC> =
    <<LC as LearnerComponentTypes>::LearningData as LearningData>::TrainOutput;

/// Type for validation output
#[allow(unused)]
pub(crate) type OutputValid<LC> =
    <<LC as LearnerComponentTypes>::LearningData as LearningData>::ValidOutput;

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

/// Concrete type that implements [training data trait](TrainingData).
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
