use crate::checkpoint::{
    AsyncCheckpointer, Checkpointer, CheckpointingAction, CheckpointingStrategy,
};
use crate::components_v2::{
    InputTrainV2, InputValidV2, LearningComponents, LearningDataV2, OutputTrainV2, OutputValidV2,
    TrainBackendV2,
};
use crate::metric::store::EventStoreClient;
use crate::{
    AsyncProcessorTraining, EventProcessorTraining, FullEventProcessorTraining,
    LearnerComponentsMarkerV2, LearningModel, TrainStep, TrainingResult, ValidStep,
};
use burn_core::module::{AutodiffModule, Module};
use burn_core::prelude::Backend;
use burn_core::tensor::Device;
use burn_core::tensor::backend::AutodiffBackend;
use burn_optim::Optimizer;
use burn_optim::lr_scheduler::LrScheduler;
use std::marker::PhantomData;

/// All components used to execute a learning paradigm, grouped in one trait.
pub trait ParadigmComponents {
    /// The data processed by the learning model during training and validation.
    type LearningData: LearningDataV2;
    /// Processes events happening during training and validation.
    type EventProcessor: EventProcessorTraining<
            ItemTrain = <Self::LearningData as LearningDataV2>::TrainOutput,
            ItemValid = <Self::LearningData as LearningDataV2>::ValidOutput,
        > + 'static;
    /// The strategy used to save and delete checkpoints.
    type CheckpointerStrategy: CheckpointingStrategy;
}

/// Concrete type that implements the [ParadigmComponents](ParadigmComponents) trait.
pub struct ParadigmComponentMarker<LD, EP, CS> {
    _learning_data: PhantomData<LD>,
    _event_processor: PhantomData<EP>,
    _strategy: PhantomData<CS>,
}

impl<LD, EP, CS> ParadigmComponents for ParadigmComponentMarker<LD, EP, CS>
where
    LD: LearningDataV2,
    EP: EventProcessorTraining<
            ItemTrain = <LD as LearningDataV2>::TrainOutput,
            ItemValid = <LD as LearningDataV2>::ValidOutput,
        > + 'static,
    CS: CheckpointingStrategy,
{
    type LearningData = LD;
    type EventProcessor = EP;
    type CheckpointerStrategy = CS;
}

/// All components used by the supervised learning paradigm, grouped in one trait.
pub trait SupervisedLearningComponents {
    /// The [ParadigmComponents](ParadigmComponents) types for supervised learning.
    type PC: ParadigmComponents<
            CheckpointerStrategy = Box<dyn CheckpointingStrategy>,
            LearningData = Self::LD,
            EventProcessor = AsyncProcessorTraining<
                FullEventProcessorTraining<OutputTrainV2<Self::LD>, OutputValidV2<Self::LD>>,
            >,
        >;
    /// The [LearningComponents](crate::LearningComponents) types for supervised learning.
    type LC: LearningComponents<Model = Self::Model, InnerModel = Self::InnerModel>;
    /// The [LearningData](crate::LearningComponents) types.
    type LD: LearningDataV2;
    /// The model to train. For supervised learning, should implement [TrainStep](crate::TrainStep).
    type Model: TrainStep<InputTrainV2<Self::LD>, OutputTrainV2<Self::LD>>
        + AutodiffModule<TrainBackendV2<Self::LC>, InnerModule = Self::InnerModel>
        + LearningModel
        + core::fmt::Display
        + 'static;
    /// The non-autodiff type of the model. For supervised learning, should implement [ValidStep](crate::TrainStep).
    type InnerModel: ValidStep<InputValidV2<Self::LD>, OutputValidV2<Self::LD>>;
}

/// Concrete type that implements the [SupervisedLearningComponents](SupervisedLearningComponents) trait.
pub struct SupervisedComponentsMarker<PC, LC, LD, M, O, S> {
    _paradigm_components: PhantomData<PC>,
    _learner_components: PhantomData<LC>,
    _learning_data: PhantomData<LD>,
    _lr_scheduler: PhantomData<S>,
    _model: PhantomData<M>,
    _optimizer: PhantomData<O>,
}

impl<PC, LC, LD, M, O, S> SupervisedLearningComponents
    for SupervisedComponentsMarker<PC, LC, LD, M, O, S>
where
    PC: ParadigmComponents<
            CheckpointerStrategy = Box<dyn CheckpointingStrategy>,
            LearningData = LD,
            EventProcessor = AsyncProcessorTraining<
                FullEventProcessorTraining<OutputTrainV2<LD>, OutputValidV2<LD>>,
            >,
        >,
    LC: LearningComponents<Model = M, InnerModel = M::InnerModule>,
    LD: LearningDataV2,
    M: TrainStep<InputTrainV2<LD>, OutputTrainV2<LD>>
        + AutodiffModule<TrainBackendV2<LC>>
        + LearningModel
        + core::fmt::Display
        + 'static,
    M::InnerModule: ValidStep<InputValidV2<LD>, OutputValidV2<LD>>,
{
    type Model = M;
    type InnerModel = M::InnerModule;
    type PC = PC;
    type LC = LC;
    type LD = LD;
}

/// The record of the learning model.
pub type LearnerModelRecord<LC> =
    <<LC as LearningComponents>::Model as Module<TrainBackendV2<LC>>>::Record;
/// The record of the optimizer.
pub type LearnerOptimizerRecord<LC> = <<LC as LearningComponents>::Optimizer as Optimizer<
    <LC as LearningComponents>::Model,
    TrainBackendV2<LC>,
>>::Record;
/// The record of the LR scheduler.
pub type LearnerSchedulerRecord<LC> =
    <<LC as LearningComponents>::LrScheduler as LrScheduler>::Record<TrainBackendV2<LC>>;

/// Provides the `run` function for any learning paradigm.
pub trait LearningParadigm<LC>
where
    LC: LearningComponents,
{
    /// Executes the learning paradigm using the provided learner.
    ///
    /// This method drives the full learning process (e.g. training loop, validation,
    /// checkpointing) and returns the final result.
    fn run(self, learner: LearnerV2<LC>) -> TrainingResult<LC::InnerModel>;
}

/// LearnerV2 struct encapsulating all components necessary to train a Neural Network model.
#[derive(Clone)]
pub struct LearnerV2<LC: LearningComponents> {
    /// The neural network model.
    pub model: LC::Model,
    /// The optimizer.
    pub optim: LC::Optimizer,
    /// The learning rate scheduler.
    pub lr_scheduler: LC::LrScheduler,
}

impl<B, LR, M, O> LearnerV2<LearnerComponentsMarkerV2<B, LR, M, O>>
where
    B: AutodiffBackend,
    LR: LrScheduler + 'static,
    M: AutodiffModule<B> + LearningModel + core::fmt::Display + 'static,
    O: Optimizer<M, B> + 'static,
{
    /// Create a learner.
    pub fn new(model: M, optim: O, lr_scheduler: LR) -> Self {
        Self {
            model,
            optim,
            lr_scheduler,
        }
    }
}

impl<LC: LearningComponents> LearnerV2<LC> {
    /// Load the module state from a [record](LearningModelRecord<LC>).
    pub fn load_model_record(&mut self, record: LearnerModelRecord<LC>) {
        self.model = self.model.clone().load_record(record);
    }

    /// Load the state of the learner's optimizer as a [record](OptimizerRecordTrain<LC>).
    pub fn load_optim_record(&mut self, record: LearnerOptimizerRecord<LC>) {
        self.optim = self.optim.clone().load_record(record);
    }

    /// Load the state of the learner's scheduler as a [record](LearnerSchedulerRecord<LC>).
    pub fn load_scheduler_record(&mut self, record: LearnerSchedulerRecord<LC>) {
        self.lr_scheduler = self.lr_scheduler.clone().load_record(record);
    }

    /// Fork the learner's model to the given device.
    pub fn fork(self, device: &<TrainBackendV2<LC> as Backend>::Device) -> Self {
        let model = self.model.fork(device);
        Self {
            model,
            optim: self.optim,
            lr_scheduler: self.lr_scheduler,
        }
    }
}

#[derive(new)]
/// Used to create, delete, or load checkpoints of the training process.
pub struct TrainingCheckpointer<LC: LearningComponents, PC: ParadigmComponents> {
    model: AsyncCheckpointer<<LC::Model as Module<LC::Backend>>::Record, LC::Backend>,
    optim: AsyncCheckpointer<
        <LC::Optimizer as Optimizer<LC::Model, LC::Backend>>::Record,
        LC::Backend,
    >,
    lr_scheduler:
        AsyncCheckpointer<<LC::LrScheduler as LrScheduler>::Record<LC::Backend>, LC::Backend>,
    strategy: PC::CheckpointerStrategy,
}

impl<LC: LearningComponents, PC: ParadigmComponents> TrainingCheckpointer<LC, PC> {
    /// Create checkpoint for the training process.
    pub fn checkpoint(&mut self, learner: &LearnerV2<LC>, epoch: usize, store: &EventStoreClient) {
        let actions = self.strategy.checkpointing(epoch, store);

        for action in actions {
            match action {
                CheckpointingAction::Delete(epoch) => {
                    self.model
                        .delete(epoch)
                        .expect("Can delete model checkpoint.");
                    self.optim
                        .delete(epoch)
                        .expect("Can delete optimizer checkpoint.");
                    self.lr_scheduler
                        .delete(epoch)
                        .expect("Can delete learning rate scheduler checkpoint.");
                }
                CheckpointingAction::Save => {
                    self.model
                        .save(epoch, learner.model.clone().into_record())
                        .expect("Can save model checkpoint.");
                    self.optim
                        .save(epoch, learner.optim.to_record())
                        .expect("Can save optimizer checkpoint.");
                    self.lr_scheduler
                        .save(epoch, learner.lr_scheduler.to_record())
                        .expect("Can save learning rate scheduler checkpoint.");
                }
            }
        }
    }

    /// Load a training checkpoint.
    pub fn load_checkpoint(
        &self,
        mut learner: LearnerV2<LC>,
        device: &Device<LC::Backend>,
        epoch: usize,
    ) -> LearnerV2<LC> {
        let record = self
            .model
            .restore(epoch, device)
            .expect("Can load model checkpoint.");
        learner.load_model_record(record);

        let record = self
            .optim
            .restore(epoch, device)
            .expect("Can load optimizer checkpoint.");
        learner.load_optim_record(record);

        let record = self
            .lr_scheduler
            .restore(epoch, device)
            .expect("Can load learning rate scheduler checkpoint.");
        learner.load_scheduler_record(record);

        learner
    }
}
