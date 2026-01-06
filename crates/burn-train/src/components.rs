use crate::{EventProcessorTraining, LearningStep, ValidStep, metric::ItemLazy};
use burn_core::{module::AutodiffModule, tensor::backend::AutodiffBackend};
use burn_optim::{Optimizer, lr_scheduler::LrScheduler};
use std::marker::PhantomData;

/// Components used for a model to learn, grouped in one trait.
pub trait LearningComponentsTypes {
    /// The backend used for learning.
    type Backend: AutodiffBackend;
    /// The learning rate scheduler used for learning.
    type LrScheduler: LrScheduler + 'static;
    /// The model that learns.
    type Model: LearningStep<LearnerInput<Self>, LearnerOutput<Self>>
        + AutodiffModule<Self::Backend, InnerModule = Self::InnerModel>
        + core::fmt::Display
        + 'static;
    /// The non-autodiff type of the model.
    type InnerModel: ValidStep<ValidInput<Self>, ValidOutput<Self>>;
    /// The optimizer used for learning.
    type Optimizer: Optimizer<Self::Model, Self::Backend> + 'static;
    /// The [LearningData](crate::LearningData) types.
    type LearningData: LearningData;
}

/// Concrete type that implements the [LearningComponentsTypes](LearningComponentsTypes) trait.
pub struct LearningComponentsMarker<B, LR, M, O, LD> {
    _backend: PhantomData<B>,
    _lr_scheduler: PhantomData<LR>,
    _model: PhantomData<M>,
    _optimizer: PhantomData<O>,
    _learning_data: PhantomData<LD>,
}

impl<B, LR, M, O, LD> LearningComponentsTypes for LearningComponentsMarker<B, LR, M, O, LD>
where
    B: AutodiffBackend,
    LR: LrScheduler + 'static,
    M: LearningStep<LD::LearningInput, LD::LearningOutput>
        + AutodiffModule<B>
        + core::fmt::Display
        + 'static,
    M::InnerModule: ValidStep<LD::ValidInput, LD::ValidOutput>,
    O: Optimizer<M, B> + 'static,
    LD: LearningData,
{
    type Backend = B;
    type LrScheduler = LR;
    type Model = M;
    type InnerModel = M::InnerModule;
    type Optimizer = O;
    type LearningData = LD;
}

/// The training backend.
pub type LearnerBackend<LC> = <LC as LearningComponentsTypes>::Backend;
/// The validation backend.
pub type ValidBackend<LC> =
    <<LC as LearningComponentsTypes>::Backend as AutodiffBackend>::InnerBackend;
/// The model of the learner.
pub type LearnerModel<LC> = <LC as LearningComponentsTypes>::Model;
/// The non-autodiff model of the learner.
pub type ValidModel<LC> = <LC as LearningComponentsTypes>::InnerModel;
/// Type for training input.
pub(crate) type LearnerInput<LC> =
    <<LC as LearningComponentsTypes>::LearningData as LearningData>::LearningInput;
/// Type for validation input.
pub(crate) type ValidInput<LC> =
    <<LC as LearningComponentsTypes>::LearningData as LearningData>::ValidInput;
/// Type for training output.
pub(crate) type LearnerOutput<LC> =
    <<LC as LearningComponentsTypes>::LearningData as LearningData>::LearningOutput;
/// Type for validation output.
pub(crate) type ValidOutput<LC> =
    <<LC as LearningComponentsTypes>::LearningData as LearningData>::ValidOutput;

/// Regroups types of input and outputs for learning and validation.
pub trait LearningData {
    /// Type of input for a step of the learning stage.
    type LearningInput: Send + 'static;
    /// Type of input for a step of the validation stage.
    type ValidInput: Send + 'static;
    /// Type of output for a step of the learning stage.
    type LearningOutput: ItemLazy + 'static;
    /// Type of output for a step of the validation stage.
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
    type LearningInput = TI;
    type ValidInput = VI;
    type LearningOutput = TO;
    type ValidOutput = VO;
}

/// All components used to execute a learning paradigm, grouped in one trait.
pub trait ParadigmComponentsTypes {
    /// The data processed by the learning model during training and validation.
    type ParadigmData: ParadigmData;
    /// Processes events happening during training and validation.
    type EventProcessor: EventProcessorTraining<
            ItemTrain = <Self::ParadigmData as ParadigmData>::LearningItem,
            ItemValid = <Self::ParadigmData as ParadigmData>::ValidItem,
        > + 'static;
}

/// Concrete type that implements the [ParadigmComponentsTypes](ParadigmComponentsTypes) trait.
pub struct ParadigmComponentsMarker<PD, EP> {
    _paradigm_data: PhantomData<PD>,
    _event_processor: PhantomData<EP>,
}

impl<PD, EP> ParadigmComponentsTypes for ParadigmComponentsMarker<PD, EP>
where
    PD: ParadigmData,
    EP: EventProcessorTraining<
            ItemTrain = <PD as ParadigmData>::LearningItem,
            ItemValid = <PD as ParadigmData>::ValidItem,
        > + 'static,
{
    type ParadigmData = PD;
    type EventProcessor = EP;
}

/// Types of items specific to the learning paradigm for learning and validation.
pub trait ParadigmData {
    /// Type of item for a step of the learning stage.
    type LearningItem: ItemLazy + 'static;
    /// Type of item for a step of the validation stage.
    type ValidItem: ItemLazy + 'static;
}

/// Concrete type that implements [ParadigmData](ParadigmData) trait.
pub struct ParadigmDataMarker<TI, VI> {
    _phantom_data: PhantomData<(TI, VI)>,
}

impl<LI, VI> ParadigmData for ParadigmDataMarker<LI, VI>
where
    LI: ItemLazy + 'static,
    VI: ItemLazy + 'static,
{
    type LearningItem = LI;
    type ValidItem = VI;
}
