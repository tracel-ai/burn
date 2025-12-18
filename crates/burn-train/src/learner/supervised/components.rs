use std::marker::PhantomData;

use burn_core::module::AutodiffModule;

use crate::{
    AsyncProcessorTraining, FullEventProcessorTraining, InputTrain, InputValid,
    LearningComponentsTypes, LearningData, OutputTrain, OutputValid, ParadigmComponentsTypes,
    TrainBackend, TrainStep, ValidStep, checkpoint::CheckpointingStrategy,
};

/// All components used by the supervised learning paradigm, grouped in one trait.
pub trait SupervisedLearningComponentsTypes {
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
        + AutodiffModule<TrainBackend<Self::LC>, InnerModule = Self::InnerModel>
        + core::fmt::Display
        + 'static;
    /// The non-autodiff type of the model. For supervised learning, should implement [ValidStep](crate::TrainStep).
    type InnerModel: ValidStep<InputValid<Self::LD>, OutputValid<Self::LD>>;
}

/// Concrete type that implements the [SupervisedLearningComponentsTypes](SupervisedLearningComponentsTypes) trait.
pub struct SupervisedLearningComponentsMarker<PC, LC, LD, M, O, S> {
    _paradigm_components: PhantomData<PC>,
    _learner_components: PhantomData<LC>,
    _learning_data: PhantomData<LD>,
    _lr_scheduler: PhantomData<S>,
    _model: PhantomData<M>,
    _optimizer: PhantomData<O>,
}

impl<PC, LC, LD, M, O, S> SupervisedLearningComponentsTypes
    for SupervisedLearningComponentsMarker<PC, LC, LD, M, O, S>
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
{
    type Model = M;
    type InnerModel = M::InnerModule;
    type PC = PC;
    type LC = LC;
    type LD = LD;
}
