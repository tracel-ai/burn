use std::{marker::PhantomData, sync::Arc};

use burn_core::data::dataloader::DataLoader;

use crate::{
    AsyncProcessorTraining, FullEventProcessorTraining, LearnerBackend, LearnerInput,
    LearnerOutput, LearningComponentsTypes, ParadigmComponentsTypes, ValidBackend, ValidInput,
    ValidOutput,
};

/// All components used by the supervised learning paradigm, grouped in one trait.
pub trait SupervisedLearningComponentsTypes {
    /// The [ParadigmComponents](ParadigmComponentsTypes) types for supervised learning.
    type PC: ParadigmComponentsTypes<
        EventProcessor = AsyncProcessorTraining<
            FullEventProcessorTraining<LearnerOutput<Self::LC>, ValidOutput<Self::LC>>,
        >,
    >;
    /// The [LearningComponents](crate::LearningComponentsTypes) types for supervised learning.
    type LC: LearningComponentsTypes;
}

/// Concrete type that implements the [SupervisedLearningComponentsTypes](SupervisedLearningComponentsTypes) trait.
pub struct SupervisedLearningComponentsMarker<PC, LC> {
    _paradigm_components: PhantomData<PC>,
    _learner_components: PhantomData<LC>,
}

impl<PC, LC> SupervisedLearningComponentsTypes for SupervisedLearningComponentsMarker<PC, LC>
where
    PC: ParadigmComponentsTypes<
        EventProcessor = AsyncProcessorTraining<
            FullEventProcessorTraining<LearnerOutput<LC>, ValidOutput<LC>>,
        >,
    >,
    LC: LearningComponentsTypes,
{
    type PC = PC;
    type LC = LC;
}

/// A reference to the training split [DataLoader](DataLoader).
pub type TrainLoader<LC> = Arc<dyn DataLoader<LearnerBackend<LC>, LearnerInput<LC>>>;
/// A reference to the validation split [DataLoader](DataLoader).
pub type ValidLoader<LC> = Arc<dyn DataLoader<ValidBackend<LC>, ValidInput<LC>>>;
