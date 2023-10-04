use crate::{checkpoint::Checkpointer, LearnerCallback};
use burn_core::{
    lr_scheduler::LrScheduler,
    module::{ADModule, Module},
    optim::Optimizer,
    tensor::backend::ADBackend,
};
use std::marker::PhantomData;

/// All components necessary to train a model grouped in one trait.
pub trait LearnerComponents {
    /// The backend in used for the training.
    type Backend: ADBackend;
    /// The learning rate scheduler used for the training.
    type LrScheduler: LrScheduler;
    /// The model to train.
    type Model: ADModule<Self::Backend> + core::fmt::Display + 'static;
    /// The optimizer used for the training.
    type Optimizer: Optimizer<Self::Model, Self::Backend>;
    /// The checkpointer used for the model.
    type CheckpointerModel: Checkpointer<<Self::Model as Module<Self::Backend>>::Record>;
    /// The checkpointer used for the optimizer.
    type CheckpointerOptimizer: Checkpointer<
        <Self::Optimizer as Optimizer<Self::Model, Self::Backend>>::Record,
    >;
    /// The checkpointer used for the scheduler.
    type CheckpointerLrScheduler: Checkpointer<<Self::LrScheduler as LrScheduler>::Record>;
    /// Callback used for training tracking.
    type Callback: LearnerCallback + 'static;
}

// pub trait DataComponents {
//     type InputTrain: Send + 'static;
//     type InputValid: Send + 'static;
//     type OutputTrain: Send + 'static;
//     type OutputValid: Send + 'static;
// }
//
// pub(crate) type Backend<T> = <<T as TrainingComponents>::Learner as LearnerComponents>::Backend;
// pub(crate) type Model<T> = <<T as TrainingComponents>::Learner as LearnerComponents>::Model;
// pub(crate) type InnerModel<T> = <<<T as TrainingComponents>::Learner as LearnerComponents>::Model as ADModule<Backend<T>>>::InnerModule;
// pub(crate) type Callback<T> = <<T as TrainingComponents>::Learner as LearnerComponents>::Callback;
// pub(crate) type InputTrain<T> = <<T as TrainingComponents>::Data as DataComponents>::InputTrain;
// pub(crate) type InputValid<T> = <<T as TrainingComponents>::Data as DataComponents>::InputValid;
// pub(crate) type OutputTrain<T> = <<T as TrainingComponents>::Data as DataComponents>::OutputTrain;
// pub(crate) type OutputValid<T> = <<T as TrainingComponents>::Data as DataComponents>::OutputValid;
//
// pub trait TrainingComponents
// where
//     Model<Self>: TrainStep<InputTrain<Self>, OutputTrain<Self>>,
//     InnerModel<Self>: ValidStep<InputValid<Self>, OutputValid<Self>>,
//     Callback<Self>: LearnerCallback<ItemTrain = OutputTrain<Self>, ItemValid = OutputValid<Self>>,
// {
//     type Learner: LearnerComponents + TrainStep<InputTrain<Self>, OutputTrain<Self>>;
//     type Data: DataComponents;
// }

/// Concrete type that implements [training components trait](TrainingComponents).
pub struct LearnerComponentsMarker<B, LR, M, O, CM, CO, CS, C> {
    _backend: PhantomData<B>,
    _lr_scheduler: PhantomData<LR>,
    _model: PhantomData<M>,
    _optimizer: PhantomData<O>,
    _checkpointer_model: PhantomData<CM>,
    _checkpointer_optim: PhantomData<CO>,
    _checkpointer_scheduler: PhantomData<CS>,
    _callback: PhantomData<C>,
}

impl<B, LR, M, O, CM, CO, CS, C> LearnerComponents
    for LearnerComponentsMarker<B, LR, M, O, CM, CO, CS, C>
where
    B: ADBackend,
    LR: LrScheduler,
    M: ADModule<B> + core::fmt::Display + 'static,
    O: Optimizer<M, B>,
    CM: Checkpointer<M::Record>,
    CO: Checkpointer<O::Record>,
    CS: Checkpointer<LR::Record>,
    C: LearnerCallback + 'static,
{
    type Backend = B;
    type LrScheduler = LR;
    type Model = M;
    type Optimizer = O;
    type CheckpointerModel = CM;
    type CheckpointerOptimizer = CO;
    type CheckpointerLrScheduler = CS;
    type Callback = C;
}
