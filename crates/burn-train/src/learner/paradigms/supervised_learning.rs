#[cfg(feature = "ddp")]
use burn_collective::CollectiveConfig;
#[cfg(feature = "ddp")]
use burn_core::tensor::backend::AutodiffBackend;
use burn_core::{data::dataloader::DataLoader, module::AutodiffModule, prelude::Backend};

use crate::{
    EarlyStoppingStrategyRef, Interrupter, ItemLazy, Learner, LearnerCheckpointer,
    LearnerComponents, LearnerSummary, LearnerV2, ParadigmComponents, ParadigmInputTrain,
    ParadigmInputValid, ParadigmOutputTrain, ParadigmOutputValid, SupervisedComponents,
    TrainLoader, TrainStep, TrainingComponents, TrainingResult, ValidLoader, ValidStep,
    checkpoint::CheckpointingStrategy,
    components_v2::{LearnerComponentTypesV2, LearningDataV2, TrainLoaderV2, ValidLoaderV2},
    metric::{
        processor::{EventProcessorTraining, LearnerEvent},
        store::EventStoreClient,
    },
    multi::CustomMultiDeviceLearningStrategy,
    single::CustomSingleDeviceLearningStrategy,
};

pub use crate::multi::MultiDeviceOptim;

type LearnerDevice<LC> = <<LC as LearnerComponentTypesV2>::Backend as Backend>::Device;

/// How should the learner run the learning for the model
#[derive(Clone)]
pub enum TrainingStrategy<LC: LearnerComponentTypesV2> {
    /// Training on one device
    SingleDevice(LearnerDevice<LC>),
    // /// Training on one device with a custom learning strategy
    // CustomSingleDevice(CustomSingleDeviceLearningStrategy<LC>),

    // /// Performs data-parallel distributed training where the optimization is
    // /// done on an elected master device.
    // MultiDevice(Vec<LearnerDevice<LC>>, MultiDeviceOptim),

    // /// Training on multiple devices with a custom learning strategy.
    // CustomMultiDevice(CustomMultiDeviceLearningStrategy<LC>),

    // /// Training with input distributed across devices, each device has its own copy of the model.
    // /// Collective ops are used to sync the gradients after each pass.
    // #[cfg(feature = "ddp")]
    // DistributedDataParallel {
    //     /// Devices on this node for the DDP
    //     devices: Vec<LearnerDevice<LC>>,

    //     /// The configuration for collective operations
    //     /// num_devices is ignored
    //     config: CollectiveConfig,
    // },
}

// /// Constructor for a distributed data parallel (DDP) learning strategy
// #[cfg(feature = "ddp")]
// pub fn ddp<B: AutodiffBackend, LC: LearnerComponentTypesV2>(
//     devices: Vec<LearnerDevice<LC>>,
//     config: CollectiveConfig,
// ) -> TrainingStrategy<LC> {
//     TrainingStrategy::DistributedDataParallel { devices, config }
// }

impl<LC: LearnerComponentTypesV2> Default for TrainingStrategy<LC> {
    fn default() -> Self {
        Self::SingleDevice(Default::default())
    }
}

/// Provides the `fit` function for any learning strategy
pub trait SupervisedLearningStrategy<SC: SupervisedComponents> {
    /// Fit the learner's model with this strategy.
    fn train(
        &self,
        mut learner: LearnerV2<SC::LC>,
        dataloader_train: TrainLoaderV2<SC::LC, SC::LD>,
        dataloader_valid: ValidLoaderV2<SC::LC, SC::LD>,
        mut training_components: TrainingComponents<SC>,
    ) -> TrainingResult<SC::InnerModel> {
        // let mut model = learner.model;
        // let mut optim = learner.optim;
        // let mut lr_scheduler = learner.lr_scheduler;

        let starting_epoch = match training_components.checkpoint {
            Some(checkpoint) => {
                if let Some(checkpointer) = &mut training_components.checkpointer {
                    // (model, optim, lr_scheduler) = checkpointer.load_checkpoint(
                    //     model,
                    //     optim,
                    //     lr_scheduler,
                    //     &Default::default(), // Load the checkpoint on the default device.
                    //     checkpoint,
                    // );
                    checkpointer.load_checkpoint(&mut learner, &Default::default(), checkpoint);
                }
                checkpoint + 1
            }
            None => 1,
        };

        // let dataloaders = self.prepare_dataloaders(dataloader_train, dataloader_valid);
        // let model = self.prepare_model(model);

        // // Training loop
        // let mut components = LearnerComponentsV2 {
        //     optim,
        //     lr_scheduler,
        //     num_epochs: training_components.num_epochs,
        //     checkpointer: training_components.checkpointer,
        //     grad_accumulation: training_components.grad_accumulation,
        //     interrupter: training_components.interrupter,
        //     early_stopping: training_components.early_stopping,
        //     event_processor: training_components.event_processor,
        //     event_store: training_components.event_store,
        // };
        // Event processor start training

        let summary_config = training_components.summary.clone();

        training_components
            .event_processor
            .process_train(LearnerEvent::Start);
        let (model, mut event_processor) = self.fit(
            training_components,
            learner,
            dataloader_train,
            dataloader_valid,
            starting_epoch,
        );

        let summary = summary_config.and_then(|summary| {
            summary
                .init()
                .map(|summary| summary.with_model(model.to_string()))
                .ok()
        });

        // Signal training end. For the TUI renderer, this handles the exit & return to main screen.
        event_processor.process_train(LearnerEvent::End(summary));

        let model = model.valid();
        let renderer = event_processor.renderer();

        TrainingResult::<SC::InnerModel> { model, renderer }
    }

    /// Training loop for this strategy
    fn fit(
        &self,
        training_components: TrainingComponents<SC>,
        learner: LearnerV2<SC::LC>,
        dataloader_train: TrainLoaderV2<SC::LC, SC::LD>,
        dataloader_valid: ValidLoaderV2<SC::LC, SC::LD>,
        starting_epoch: usize,
    ) -> (SC::Model, <SC::PC as ParadigmComponents>::EventProcessor);
}

// /// Struct to minimise parameters passed to [SupervisedLearningStrategy::learn]
// /// These components are used during training
// pub struct LearnerComponentsV2<LC: LearnerComponentTypesV2> {
//     /// The [Optimizer](LearnerComponentTypesV2::Optimizer) used for the training.
//     pub optim: LC::Optimizer,
//     /// The [learning rate scheduler](LearnerComponentTypesV2::LrScheduler) used for the training.
//     pub lr_scheduler: LC::LrScheduler,
//     /// The number of epochs the training should last.
//     pub num_epochs: usize,
//     /// Enables gradients accumulation.
//     pub grad_accumulation: Option<usize>,
//     /// A [LearnerCheckpointer](LearnerCheckpointer) used to save and load training checkpoints.
//     pub checkpointer: Option<LearnerCheckpointerV2<LC>>,
//     /// An [Interupter](Interrupter) that allows aborting the training/evaluation process early.
//     pub interrupter: Interrupter,
//     /// Cloneable reference to an early stopping strategy.
//     pub early_stopping: Option<EarlyStoppingStrategyRef>,
//     /// An [EventProcessor](LearnerComponentTypesV2::EventProcessor) that processes events happening during training and validation.
//     pub event_processor: LC::EventProcessor,
//     /// A reference to an [EventStoreClient](EventStoreClient).
//     pub event_store: Arc<EventStoreClient>,
// }
