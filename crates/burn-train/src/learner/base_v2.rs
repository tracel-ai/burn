use crate::checkpoint::{
    AsyncCheckpointer, Checkpointer, CheckpointingAction, CheckpointingStrategy,
};
use crate::components::TrainBackend;
use crate::components_v2::{
    InputTrainV2, InputValidV2, LearnerComponentTypesV2, LearningDataV2, OutputTrainV2,
    OutputValidV2, TrainBackendV2,
};
// use crate::learner::paradigms::{
//     SingleDeviceLearningStrategyV2, SupervisedLearningStrategy, TrainLoaderV2, TrainingStrategy,
//     ValidLoaderV2,
// };
use crate::metric::store::EventStoreClient;
use crate::{
    AsyncProcessorTraining, CloneEarlyStoppingStrategy, EventProcessorTraining,
    FullEventProcessorTraining, Interrupter, LearnerSummaryConfig, LearningModel, LearningStrategy,
    TrainLoader, TrainStep, TrainingResult, ValidStep,
};
use burn_core::module::{AutodiffModule, Module};
use burn_core::prelude::Backend;
use burn_core::tensor::Device;
use burn_core::tensor::backend::AutodiffBackend;
use burn_optim::lr_scheduler::LrScheduler;
use burn_optim::{GradientsParams, Optimizer};
use std::marker::PhantomData;
use std::sync::Arc;

pub trait ParadigmComponents {
    type LearningData: LearningDataV2;
    type EventProcessor: EventProcessorTraining<
            ItemTrain = <Self::LearningData as LearningDataV2>::TrainOutput,
            ItemValid = <Self::LearningData as LearningDataV2>::ValidOutput,
        > + 'static;
    type CheckpointerStrategy: CheckpointingStrategy;
}

pub struct ParadigmComponentMarker<LD, EP, CS> {
    // pub struct LearnerComponentsMarkerV2<B, LR, M, O> {
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

pub trait SupervisedComponents {
    type PC: ParadigmComponents<
            CheckpointerStrategy = Box<dyn CheckpointingStrategy>,
            LearningData = Self::LD,
            EventProcessor = AsyncProcessorTraining<
                FullEventProcessorTraining<OutputTrainV2<Self::LD>, OutputValidV2<Self::LD>>,
            >,
        >;
    type LC: LearnerComponentTypesV2<
            Backend = Self::Backend,
            Model = Self::Model,
            InnerModel = Self::InnerModel,
            CheckpointerModel = AsyncCheckpointer<
                <Self::Model as Module<Self::Backend>>::Record,
                Self::Backend,
            >,
            CheckpointerOptimizer = AsyncCheckpointer<
                <Self::Optimizer as Optimizer<Self::Model, Self::Backend>>::Record,
                Self::Backend,
            >,
            CheckpointerLrScheduler = AsyncCheckpointer<
                <Self::LrScheduler as LrScheduler>::Record<Self::Backend>,
                Self::Backend,
            >,
        >;
    type LD: LearningDataV2;
    type Model: TrainStep<InputTrainV2<Self::LD>, OutputTrainV2<Self::LD>>
        + AutodiffModule<Self::Backend, InnerModule = Self::InnerModel>
        + LearningModel
        + core::fmt::Display
        + 'static;
    type InnerModel: ValidStep<InputValidV2<Self::LD>, OutputValidV2<Self::LD>>;
    type Optimizer: Optimizer<Self::Model, Self::Backend>;
    type LrScheduler: LrScheduler;
    type Backend: AutodiffBackend;
}

pub struct SupervisedComponentsMarkerV2<PC, LC, LD, M, O, S, B> {
    // pub struct LearnerComponentsMarkerV2<B, LR, M, O> {
    _paradigm_components: PhantomData<PC>,
    _learner_components: PhantomData<LC>,
    _learning_data: PhantomData<LD>,
    _backend: PhantomData<B>,
    _lr_scheduler: PhantomData<S>,
    _model: PhantomData<M>,
    _optimizer: PhantomData<O>,
}

impl<PC, LC, LD, M, O, S, B> SupervisedComponents
    for SupervisedComponentsMarkerV2<PC, LC, LD, M, O, S, B>
where
    PC: ParadigmComponents<
            CheckpointerStrategy = Box<dyn CheckpointingStrategy>,
            LearningData = LD,
            EventProcessor = AsyncProcessorTraining<
                FullEventProcessorTraining<OutputTrainV2<LD>, OutputValidV2<LD>>,
            >,
        >,
    LC: LearnerComponentTypesV2<
            Backend = B,
            Model = M,
            InnerModel = M::InnerModule,
            CheckpointerModel = AsyncCheckpointer<<M as Module<B>>::Record, B>,
            CheckpointerOptimizer = AsyncCheckpointer<<O as Optimizer<M, B>>::Record, B>,
            CheckpointerLrScheduler = AsyncCheckpointer<<S as LrScheduler>::Record<B>, B>,
        >,
    LD: LearningDataV2,
    M: TrainStep<InputTrainV2<LD>, OutputTrainV2<LD>>
        + AutodiffModule<B>
        + LearningModel
        + core::fmt::Display
        + 'static,
    M::InnerModule: ValidStep<InputValidV2<LD>, OutputValidV2<LD>>,
    O: Optimizer<M, B>,
    S: LrScheduler,
    B: AutodiffBackend,
{
    type Backend = B;
    type LrScheduler = S;
    type Model = M;
    type InnerModel = M::InnerModule;
    type Optimizer = O;
    type PC = PC;
    type LC = LC;
    type LD = LD;
    // type CheckpointerStrategy = S;
}
// pub type ParadigmLearnerComponents<PC> = <PC as ParadigmComponents>::LearnerComponents;
// pub type ParadigmLearningData<PC> = <PC as ParadigmComponents>::LearningData;
// pub type ParadigmModel<PC> =
//     <<PC as ParadigmComponents>::LearnerComponents as LearnerComponentTypesV2>::Model;
// pub type ParadigmInnerModel<PC> =
//     <<PC as ParadigmComponents>::LearnerComponents as LearnerComponentTypesV2>::InnerModel;
// pub type ParadigmOptimizer<PC> =
//     <<PC as ParadigmComponents>::LearnerComponents as LearnerComponentTypesV2>::Optimizer;
// pub type ParadigmScheduler<PC> =
//     <<PC as ParadigmComponents>::LearnerComponents as LearnerComponentTypesV2>::LrScheduler;
// pub type ParadigmModelCheckpointer<PC> =
//     <<PC as ParadigmComponents>::LearnerComponents as LearnerComponentTypesV2>::CheckpointerModel;
// pub type ParadigmOptimizerCheckpointer<PC> =
//     <<PC as ParadigmComponents>::LearnerComponents as LearnerComponentTypesV2>::CheckpointerOptimizer;
// pub type ParadigmSchedulerCheckpointer<PC> =
//     <<PC as ParadigmComponents>::LearnerComponents as LearnerComponentTypesV2>::CheckpointerLrScheduler;
// pub type ParadigmBackendTrain<PC> =
//     <<PC as ParadigmComponents>::LearnerComponents as LearnerComponentTypesV2>::Backend;
// pub type ParadigmBackendValid<PC> =
//     <<<PC as ParadigmComponents>::LearnerComponents as LearnerComponentTypesV2>::Backend as AutodiffBackend>::InnerBackend;
pub type ParadigmInputTrain<PC> =
    <<PC as ParadigmComponents>::LearningData as LearningDataV2>::TrainInput;
pub type ParadigmOutputTrain<PC> =
    <<PC as ParadigmComponents>::LearningData as LearningDataV2>::TrainOutput;
pub type ParadigmInputValid<PC> =
    <<PC as ParadigmComponents>::LearningData as LearningDataV2>::ValidInput;
pub type ParadigmOutputValid<PC> =
    <<PC as ParadigmComponents>::LearningData as LearningDataV2>::ValidOutput;
pub type ModelRecordTrain<LC> =
    <<LC as LearnerComponentTypesV2>::Model as Module<TrainBackendV2<LC>>>::Record;
pub type OptimizerRecordTrain<LC> = <<LC as LearnerComponentTypesV2>::Optimizer as Optimizer<
    <LC as LearnerComponentTypesV2>::Model,
    TrainBackendV2<LC>,
>>::Record;
pub type SchedulerRecordTrain<LC> =
    <<LC as LearnerComponentTypesV2>::LrScheduler as LrScheduler>::Record<TrainBackendV2<LC>>;

// pub trait LearningParadigm<LC>
// where
//     LC: LearnerComponentTypesV2,
// {
//     fn train(self, learner: LearnerV2<LC>) -> TrainingResult<LC::InnerModel>;
// }

pub trait LearningParadigm<LC>
where
    LC: LearnerComponentTypesV2,
{
    fn train(self) -> TrainingResult<LC::InnerModel>;
}

// pub struct SupervisedTraining<LC: LearnerComponentTypesV2> {
//     dataloader_train: TrainLoaderV2<LC>,
//     dataloader_valid: ValidLoaderV2<LC>,
//     strategy: TrainingStrategy<LC>,
//     // pub(crate) num_epochs: usize,
//     // pub(crate) checkpoint: Option<usize>,
//     // pub(crate) grad_accumulation: Option<usize>,
//     // pub(crate) checkpointer: Option<TrainingCheckpointer<LC>>,
//     // pub(crate) interrupter: Interrupter,
//     // pub(crate) early_stopping: Option<EarlyStoppingStrategyRefV2>,
//     // pub(crate) event_processor: LC::EventProcessor,
//     // pub(crate) event_store: Arc<EventStoreClient>,
//     // pub(crate) summary:
// }

// impl<LC: LearnerComponentTypesV2> SupervisedTraining<LC> {
//     pub fn new(
//         dataloader_train: TrainLoaderV2<LC>,
//         dataloader_valid: ValidLoaderV2<LC>,
//         strategy: TrainingStrategy<LC>,
//     ) -> Self {
//         Self {
//             dataloader_train,
//             dataloader_valid,
//             strategy,
//         }
//     }
// }

// impl<LC: LearnerComponentTypesV2> LearningParadigm<LC> for SupervisedTraining<LC>
// where
//     LC::Model: TrainStep<
//             <LC::LearningDataV2 as LearningDataV2>::TrainInput,
//             <LC::LearningDataV2 as LearningDataV2>::TrainOutput,
//         >,
// {
//     fn train(self, learner: LearnerV2<LC>) {
//         match self.strategy {
//             TrainingStrategy::SingleDevice(device) => {
//                 let single_device = SingleDeviceLearningStrategyV2::new(device);
//                 single_device.fit(learner, self.dataloader_train, self.dataloader_valid);
//             }
//         }

//         // self.fit(
//         //     learner,
//         //     self.dataloader_train.clone(),
//         //     self.dataloader_valid.clone(),
//         //     self.strategy,
//         // )
//     }
// }

/// LearnerV2 struct encapsulating all components necessary to train a Neural Network model.
///
/// To create a learner, use the [builder](crate::learner::LearnerBuilder) struct.
pub struct LearnerV2<LC: LearnerComponentTypesV2> {
    pub model: LC::Model,
    pub optim: LC::Optimizer,
    pub lr_scheduler: LC::LrScheduler,
    // pub(crate) num_epochs: usize,
    // pub(crate) checkpoint: Option<usize>,
    // pub(crate) grad_accumulation: Option<usize>,
    // pub(crate) checkpointer: Option<TrainingCheckpointer<LC>>,
    // pub(crate) interrupter: Interrupter,
    // pub(crate) early_stopping: Option<EarlyStoppingStrategyRefV2>,
    // pub(crate) event_processor: LC::EventProcessor,
    // pub(crate) event_store: Arc<EventStoreClient>,
    // pub(crate) summary: Option<LearnerSummaryConfig>,
}

impl<LC: LearnerComponentTypesV2> LearnerV2<LC> {
    pub fn load_model_record(&mut self, record: ModelRecordTrain<LC>) {
        self.model = self.model.clone().load_record(record);
    }

    pub fn load_optim_record(&mut self, record: OptimizerRecordTrain<LC>) {
        self.optim = self.optim.clone().load_record(record);
    }

    pub fn load_scheduler_record(&mut self, record: SchedulerRecordTrain<LC>) {
        self.lr_scheduler = self.lr_scheduler.clone().load_record(record);
    }

    pub fn fork(self, device: &<TrainBackendV2<LC> as Backend>::Device) -> Self {
        let model = self.model.fork(device);
        Self {
            model,
            optim: self.optim,
            lr_scheduler: self.lr_scheduler,
        }
    }
}

/// Cloneable reference to an early stopping strategy
pub(crate) type EarlyStoppingStrategyRefV2 = Box<dyn CloneEarlyStoppingStrategy>;

// #[derive(new)]
// /// Used to create, delete, or load checkpoints of the training process.
// pub struct TrainingCheckpointer<PC: ParadigmComponents> {
//     model: ParadigmModelCheckpointer<PC>,
//     optim: ParadigmOptimizerCheckpointer<PC>,
//     lr_scheduler: ParadigmSchedulerCheckpointer<PC>,
//     strategy: PC::CheckpointerStrategy,
// }

#[derive(new)]
/// Used to create, delete, or load checkpoints of the training process.
pub struct TrainingCheckpointer<LC: LearnerComponentTypesV2, PC: ParadigmComponents> {
    model: LC::CheckpointerModel,
    optim: LC::CheckpointerOptimizer,
    lr_scheduler: LC::CheckpointerLrScheduler,
    strategy: PC::CheckpointerStrategy,
}

// impl<LC: LearnerComponentTypesV2> TrainingCheckpointer<LC> {
//     pub fn new(
//         model: LC::CheckpointerModel,
//         optim: LC::CheckpointerOptimizer,
//         lr_scheduler: LC::CheckpointerLrScheduler,
//         strategy: Box<dyn CheckpointingStrategy>,
//     ) -> Self {
//         Self {
//             model,
//             optim,
//             lr_scheduler,
//             strategy,
//         }
//     }
// }

impl<LC: LearnerComponentTypesV2, PC: ParadigmComponents> TrainingCheckpointer<LC, PC> {
    /// Create checkpoint for the training process.
    pub fn checkpoint(
        &mut self,
        model: &LC::Model,
        optim: &LC::Optimizer,
        lr_scheduler: &LC::LrScheduler,
        epoch: usize,
        store: &EventStoreClient,
    ) {
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
                        .save(epoch, model.clone().into_record())
                        .expect("Can save model checkpoint.");
                    self.optim
                        .save(epoch, optim.to_record())
                        .expect("Can save optimizer checkpoint.");
                    self.lr_scheduler
                        .save(epoch, lr_scheduler.to_record())
                        .expect("Can save learning rate scheduler checkpoint.");
                }
            }
        }
    }

    /// Load a training checkpoint.
    pub fn load_checkpoint(
        &self,
        learner: &mut LearnerV2<LC>,
        // model: ParadigmModel<PC>,
        // optim: ParadigmOptimizer<PC>,
        // lr_scheduler: ParadigmScheduler<PC>,
        device: &Device<LC::Backend>,
        epoch: usize,
    )
    // ->(
    //     ParadigmModel<PC>,
    //     ParadigmOptimizer<PC>,
    //     ParadigmScheduler<PC>,
    // )
    {
        let record = self
            .model
            .restore(epoch, device)
            .expect("Can load model checkpoint.");
        // let model = model.load_record(record);
        learner.load_model_record(record);

        let record = self
            .optim
            .restore(epoch, device)
            .expect("Can load optimizer checkpoint.");
        // let optim = optim.load_record(record);
        learner.load_optim_record(record);

        let record = self
            .lr_scheduler
            .restore(epoch, device)
            .expect("Can load learning rate scheduler checkpoint.");
        // let scheduler = lr_scheduler.load_record(record);
        learner.load_scheduler_record(record);

        // (model, optim, scheduler)
    }
}

// /// todo
// pub struct LearnerV2<LC: LearnerComponentTypesV2> {
//     /// todo
//     pub model: LC::Model,
//     /// todo
//     pub optim: LC::Optimizer,
//     /// todo
//     pub lr_scheduler: LC::LrScheduler,
//     // pub(crate) num_epochs: usize,
//     // pub(crate) checkpoint: Option<usize>,
//     // pub(crate) grad_accumulation: Option<usize>,
//     // /// todo
//     // pub checkpointer: Option<TrainingCheckpointer<LC>>,
//     // pub(crate) execution_strategy: ExecutionStrategy<LC>,
//     // pub(crate) learning_paradigm: LP,
//     // pub interrupter: Interrupter,
//     // pub(crate) early_stopping: Option<EarlyStoppingStrategyRef>,
// }

// // /// LearnerV2 struct encapsulating all components nec
// // /// Cloneable reference to an early stopping strategy
// // pub(crate) type EarlyStoppingStrategyRef = Box<dyn CloneEarlyStoppingStrategy>;

// #[derive(new)]
// /// Used to create, delete, or load checkpoints of the training process.
// pub struct TrainingCheckpointer<LC: LearnerComponentTypesV2, S: CheckpointingStrategy> {
//     model: <LC::Model as Module<LC::Backend>>::Record,
//     optim: LC::Optimizer,
//     lr_scheduler: LC::LrScheduler,
//     strategy: S,
// }

// impl<LC: LearnerComponentTypesV2, S: CheckpointingStrategy> TrainingCheckpointer<LC, S> {
//     /// Create checkpoint for the training process.
//     pub fn checkpoint(
//         &mut self,
//         model: &LC::Model,
//         optim: &LC::Optimizer,
//         scheduler: &LC::LrScheduler,
//         epoch: usize,
//         store: &EventStoreClient,
//     ) {
//         let actions = self.strategy.checkpointing(epoch, store);

//         for action in actions {
//             match action {
//                 CheckpointingAction::Delete(epoch) => {
//                     self.model
//                         .delete(epoch)
//                         .expect("Can delete model checkpoint.");
//                     self.optim
//                         .delete(epoch)
//                         .expect("Can delete optimizer checkpoint.");
//                     self.lr_scheduler
//                         .delete(epoch)
//                         .expect("Can delete learning rate scheduler checkpoint.");
//                 }
//                 CheckpointingAction::Save => {
//                     self.model
//                         .save(epoch, model.clone().into_record())
//                         .expect("Can save model checkpoint.");
//                     self.optim
//                         .save(epoch, optim.to_record())
//                         .expect("Can save optimizer checkpoint.");
//                     self.lr_scheduler
//                         .save(epoch, scheduler.to_record())
//                         .expect("Can save learning rate scheduler checkpoint.");
//                 }
//             }
//         }
//     }

//     /// Load a training checkpoint.
//     pub fn load_checkpoint(
//         &self,
//         model: LC::Model,
//         optim: LC::Optimizer,
//         scheduler: LC::LrScheduler,
//         device: &Device<LC::Backend>,
//         epoch: usize,
//     ) -> (LC::Model, LC::Optimizer, LC::LrScheduler) {
//         let record = self
//             .model
//             .restore(epoch, device)
//             .expect("Can load model checkpoint.");
//         let model = model.load_record(record);

//         let record = self
//             .optim
//             .restore(epoch, device)
//             .expect("Can load optimizer checkpoint.");
//         let optim = optim.load_record(record);

//         let record = self
//             .lr_scheduler
//             .restore(epoch, device)
//             .expect("Can load learning rate scheduler checkpoint.");
//         let scheduler = scheduler.load_record(record);

//         (model, optim, scheduler)
//     }
// }

// #[derive(Clone, Default)]
// /// A handle that allows aborting the training/evaluation process early.
// pub struct Interrupter {
//     state: Arc<AtomicBool>,
// }

// impl Interrupter {
//     /// Create a new instance.
//     pub fn new() -> Self {
//         Self::default()
//     }

//     /// Notify the learner that it should stop.
//     pub fn stop(&self) {
//         self.state.store(true, Ordering::Relaxed);
//     }

//     /// Reset the interrupter.
//     pub fn reset(&self) {
//         self.state.store(false, Ordering::Relaxed);
//     }

//     /// True if .stop() has been called.
//     pub fn should_stop(&self) -> bool {
//         self.state.load(Ordering::Relaxed)
//     }
// }
