use std::marker::PhantomData;

use burn_core::module::AutodiffModule;
use burn_rl::{Environment, LearningAgent};

use crate::{
    AsyncProcessorTraining, FullEventProcessorTraining, InputTrain, InputValid,
    LearningComponentsTypes, LearningData, OutputTrain, OutputValid, ParadigmComponentsTypes,
    TrainBackend, TrainStep, ValidStep, checkpoint::CheckpointingStrategy,
};

/// All components used by the supervised learning paradigm, grouped in one trait.
pub trait OffPolicyLearningComponentsTypes {
    /// The [ParadigmComponents](ParadigmComponentsTypes) types for supervised learning.
    type PC: ParadigmComponentsTypes<
            CheckpointerStrategy = Box<dyn CheckpointingStrategy>,
            LearningData = Self::LD,
            EventProcessor = AsyncProcessorTraining<
                FullEventProcessorTraining<OutputTrain<Self::LD>, OutputValid<Self::LD>>,
            >,
        >;
    /// The [LearningComponents](crate::LearningComponentsTypes) types for supervised learning.
    type LC: LearningComponentsTypes<Model = Self::Model, InnerModel = Self::InnerModel>;
    /// The [LearningData](crate::LearningData) types.
    type LD: LearningData;
    /// The model to train. For supervised learning, should implement [TrainStep](crate::TrainStep).
    type Model: TrainStep<InputTrain<Self::LD>, OutputTrain<Self::LD>>
        // TODO : Instead of Train/ValidStep, do offpolicymodel trait that takes as input a transition (generic over env) and outputs logits.
        + AutodiffModule<TrainBackend<Self::LC>, InnerModule = Self::InnerModel>
        + core::fmt::Display
        + 'static;
    /// The non-autodiff type of the model. For supervised learning, should implement [ValidStep](crate::TrainStep).
    type InnerModel: ValidStep<InputValid<Self::LD>, OutputValid<Self::LD>>;
    type Env: Environment;
    type Agent: LearningAgent<TrainBackend<Self::LC>, Self::Env>;
}

/// Concrete type that implements the [OffPolicyLearningComponentsTypes](OffPolicyLearningComponentsTypes) trait.
pub struct OffPolicyLearningComponentsMarker<PC, LC, LD, M, O, S, E, A> {
    _paradigm_components: PhantomData<PC>,
    _learner_components: PhantomData<LC>,
    _learning_data: PhantomData<LD>,
    _lr_scheduler: PhantomData<S>,
    _model: PhantomData<M>,
    _optimizer: PhantomData<O>,
    _env: PhantomData<E>,
    _agent: PhantomData<A>,
}

impl<PC, LC, LD, M, O, S, E, A> OffPolicyLearningComponentsTypes
    for OffPolicyLearningComponentsMarker<PC, LC, LD, M, O, S, E, A>
where
    PC: ParadigmComponentsTypes<
            CheckpointerStrategy = Box<dyn CheckpointingStrategy>,
            LearningData = LD,
            EventProcessor = AsyncProcessorTraining<
                FullEventProcessorTraining<OutputTrain<LD>, OutputValid<LD>>,
            >,
        >,
    LC: LearningComponentsTypes<Model = M, InnerModel = M::InnerModule>,
    LD: LearningData,
    M: TrainStep<InputTrain<LD>, OutputTrain<LD>>
        + AutodiffModule<TrainBackend<LC>>
        + core::fmt::Display
        + 'static,
    M::InnerModule: ValidStep<InputValid<LD>, OutputValid<LD>>,
    E: Environment,
    A: LearningAgent<TrainBackend<LC>, E>,
{
    type Model = M;
    type InnerModel = M::InnerModule;
    type PC = PC;
    type LC = LC;
    type LD = LD;
    type Env = E;
    type Agent = A;
}
