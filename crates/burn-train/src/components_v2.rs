use crate::{
    EventProcessorTraining, LearningModel,
    checkpoint::{Checkpointer, CheckpointingStrategy},
    metric::ItemLazy,
};
use burn_core::{
    data::dataloader::DataLoader,
    module::{AutodiffModule, Module},
    tensor::backend::AutodiffBackend,
};
use burn_optim::{Optimizer, lr_scheduler::LrScheduler};
use std::{marker::PhantomData, sync::Arc};

// /// All components necessary to train a model grouped in one trait.
// pub trait LearnerComponentTypesV2 {
//     /// The backend used for the training.
//     type Backend: AutodiffBackend;
//     /// The learning rate scheduler used for the training.
//     type LrScheduler: LrScheduler;
//     /// The model to train.
//     type Model: AutodiffModule<Self::Backend, InnerModule = Self::InnerModel>
//         + core::fmt::Display
//         + 'static;
//     /// The non-autodiff type of the model, should implement ValidationStep
//     type InnerModel;
//     /// The optimizer used for the training.
//     type Optimizer: Optimizer<Self::Model, Self::Backend>;
//     /// The checkpointer used for the model.
//     type CheckpointerModel: Checkpointer<<Self::Model as Module<Self::Backend>>::Record, Self::Backend>;
//     /// The checkpointer used for the optimizer.
//     type CheckpointerOptimizer: Checkpointer<
//             <Self::Optimizer as Optimizer<Self::Model, Self::Backend>>::Record,
//             Self::Backend,
//         > + Send;
//     /// The checkpointer used for the scheduler.
//     type CheckpointerLrScheduler: Checkpointer<<Self::LrScheduler as LrScheduler>::Record<Self::Backend>, Self::Backend>;
//     /// Processes events happening during training and valid.
//     type EventProcessor: EventProcessorTraining<
//             ItemTrain = <Self::LearningDataV2 as LearningDataV2>::TrainOutput,
//             ItemValid = <Self::LearningDataV2 as LearningDataV2>::ValidOutput,
//         > + 'static;
//     /// The strategy to save and delete checkpoints.
//     type CheckpointerStrategy: CheckpointingStrategy;
//     /// The data used to perform training and validation.
//     type LearningDataV2: LearningDataV2;
// }

// /// Concrete type that implements [training components trait](TrainingComponents).
// pub struct LearnerComponentsMarkerV2<B, LR, M, O, CM, CO, CS, EP, S, LD> {
//     _backend: PhantomData<B>,
//     _lr_scheduler: PhantomData<LR>,
//     _model: PhantomData<M>,
//     _optimizer: PhantomData<O>,
//     _checkpointer_model: PhantomData<CM>,
//     _checkpointer_optim: PhantomData<CO>,
//     _checkpointer_scheduler: PhantomData<CS>,
//     _event_processor: PhantomData<EP>,
//     _strategy: S,
//     _learning_data: PhantomData<LD>,
// }

// impl<B, LR, M, O, CM, CO, CS, EP, S, LD> LearnerComponentTypesV2
//     for LearnerComponentsMarkerV2<B, LR, M, O, CM, CO, CS, EP, S, LD>
// where
//     B: AutodiffBackend,
//     LR: LrScheduler,
//     M: AutodiffModule<B> + core::fmt::Display + 'static,
//     O: Optimizer<M, B>,
//     CM: Checkpointer<M::Record, B>,
//     CO: Checkpointer<O::Record, B>,
//     CS: Checkpointer<LR::Record<B>, B>,
//     EP: EventProcessorTraining<ItemTrain = LD::TrainOutput, ItemValid = LD::ValidOutput> + 'static,
//     S: CheckpointingStrategy,
//     LD: LearningDataV2,
// {
//     type Backend = B;
//     type LrScheduler = LR;
//     type Model = M;
//     type InnerModel = M::InnerModule;
//     type Optimizer = O;
//     type CheckpointerModel = CM;
//     type CheckpointerOptimizer = CO;
//     type CheckpointerLrScheduler = CS;
//     type EventProcessor = EP;
//     type CheckpointerStrategy = S;
//     type LearningDataV2 = LD;
// }

// /// The training backend.
// pub type TrainBackendV2<LC> = <LC as LearnerComponentTypesV2>::Backend;

// /// The validation backend.
// pub type ValidBackendV2<LC> =
//     <<LC as LearnerComponentTypesV2>::Backend as AutodiffBackend>::InnerBackend;

// /// Type for training input
// pub(crate) type InputTrainV2<LC> =
//     <<LC as LearnerComponentTypesV2>::LearningDataV2 as LearningDataV2>::TrainInput;

// /// Type for validation input
// pub(crate) type InputValidV2<LC> =
//     <<LC as LearnerComponentTypesV2>::LearningDataV2 as LearningDataV2>::ValidInput;

// /// Type for training output
// pub(crate) type OutputTrainV2<LC> =
//     <<LC as LearnerComponentTypesV2>::LearningDataV2 as LearningDataV2>::TrainOutput;

// /// Type for validation output
// #[allow(unused)]
// pub(crate) type OutputValidV2<LC> =
//     <<LC as LearnerComponentTypesV2>::LearningDataV2 as LearningDataV2>::ValidOutput;

pub trait LearnerComponentTypesV2 {
    /// The backend used for the training.
    type Backend: AutodiffBackend;
    /// The learning rate scheduler used for the training.
    type LrScheduler: LrScheduler;
    /// The model to train.
    type Model: AutodiffModule<Self::Backend, InnerModule = Self::InnerModel>
        + LearningModel
        + core::fmt::Display
        + 'static;
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
    // /// The strategy to save and delete checkpoints.
    // type CheckpointerStrategy: CheckpointingStrategy;
    // /// The data used to perform training and validation.
    // type LearningDataV2: LearningDataV2;
}

pub struct LearnerComponentsMarkerV2<B, LR, M, O, CM, CO, CS> {
    // pub struct LearnerComponentsMarkerV2<B, LR, M, O> {
    _backend: PhantomData<B>,
    _lr_scheduler: PhantomData<LR>,
    _model: PhantomData<M>,
    _optimizer: PhantomData<O>,
    _checkpointer_model: PhantomData<CM>,
    _checkpointer_optim: PhantomData<CO>,
    _checkpointer_scheduler: PhantomData<CS>,
    // _strategy: S,
}

impl<B, LR, M, O, CM, CO, CS> LearnerComponentTypesV2
    for LearnerComponentsMarkerV2<B, LR, M, O, CM, CO, CS>
// impl<B, LR, M, O> LearnerComponentTypesV2 for LearnerComponentsMarkerV2<B, LR, M, O>
where
    B: AutodiffBackend,
    LR: LrScheduler,
    M: AutodiffModule<B> + LearningModel + core::fmt::Display + 'static,
    O: Optimizer<M, B>,
    CM: Checkpointer<M::Record, B>,
    CO: Checkpointer<O::Record, B>,
    CS: Checkpointer<LR::Record<B>, B>,
    // S: CheckpointingStrategy,
{
    type Backend = B;
    type LrScheduler = LR;
    type Model = M;
    type InnerModel = M::InnerModule;
    type Optimizer = O;
    type CheckpointerModel = CM;
    type CheckpointerOptimizer = CO;
    type CheckpointerLrScheduler = CS;
    // type CheckpointerStrategy = S;
}

/// The training backend.
pub type TrainBackendV2<LC> = <LC as LearnerComponentTypesV2>::Backend;
/// The validation backend.
pub type ValidBackendV2<LC> =
    <<LC as LearnerComponentTypesV2>::Backend as AutodiffBackend>::InnerBackend;
/// Type for training input
pub(crate) type InputTrainV2<LD> = <LD as LearningDataV2>::TrainInput;
/// Type for validation input
pub(crate) type InputValidV2<LD> = <LD as LearningDataV2>::ValidInput;
/// Type for training output
pub(crate) type OutputTrainV2<LD> = <LD as LearningDataV2>::TrainOutput;
/// Type for validation output
#[allow(unused)]
pub(crate) type OutputValidV2<LD> = <LD as LearningDataV2>::ValidOutput;
/// A reference to the training split [DataLoader](DataLoader).
pub type TrainLoaderV2<LC, LD> = Arc<dyn DataLoader<TrainBackendV2<LC>, InputTrainV2<LD>>>;
/// A reference to the validation split [DataLoader](DataLoader).
pub type ValidLoaderV2<LC, LD> = Arc<dyn DataLoader<ValidBackendV2<LC>, InputValidV2<LD>>>;
pub type TrainModelV2<LC> = <LC as LearnerComponentTypesV2>::Model;
pub type TrainOptmizerV2<LC> = <LC as LearnerComponentTypesV2>::Optimizer;
pub type TrainSchedulerV2<LC> = <LC as LearnerComponentTypesV2>::LrScheduler;

/// Regroups types of input and outputs for training and validation
pub trait LearningDataV2 {
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
pub struct LearningDataMarkerV2<TI, VI, TO, VO> {
    _phantom_data: PhantomData<(TI, VI, TO, VO)>,
}

impl<TI, VI, TO, VO> LearningDataV2 for LearningDataMarkerV2<TI, VI, TO, VO>
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
