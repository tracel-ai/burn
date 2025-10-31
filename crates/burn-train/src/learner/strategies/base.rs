use std::sync::Arc;

#[cfg(feature = "ddp")]
use burn_collective::CollectiveConfig;
use burn_core::{module::AutodiffModule, prelude::Backend, tensor::backend::AutodiffBackend};

use crate::{
    EarlyStoppingStrategyRef, Interrupter, Learner, LearnerCheckpointer, TrainLoader,
    TrainingResult, ValidLoader,
    components::LearnerComponentTypes,
    metric::{
        processor::{EventProcessorTraining, LearnerEvent},
        store::EventStoreClient,
    },
    multi::CustomMultiDeviceLearningStrategy,
    single::CustomSingleDeviceLearningStrategy,
};

type LearnerDevice<LC> = <<LC as LearnerComponentTypes>::Backend as Backend>::Device;

/// How should the learner run the learning for the model
#[derive(Clone)]
pub enum LearningStrategy<LC: LearnerComponentTypes> {
    /// Training on one device
    SingleDevice(LearnerDevice<LC>),

    /// Training on one device with a custom learning strategy
    CustomSingleDevice(CustomSingleDeviceLearningStrategy<LC>),

    /// Legacy implementation of local multi-device training
    MultiDeviceNaive(Vec<LearnerDevice<LC>>),

    /// Training on multiple devices with a custom learning strategy.
    CustomMultiDevice(CustomMultiDeviceLearningStrategy<LC>),

    /// Training with input distributed across devices, each device has its own copy of the model.
    /// Collective ops are used to sync the gradients after each pass.
    #[cfg(feature = "ddp")]
    DistributedDataParallel {
        /// Devices on this node for the DDP
        devices: Vec<LearnerDevice<LC>>,

        /// The configuration for collective operations
        /// num_devices is ignored
        config: CollectiveConfig,
    },
}

/// Constructor for a distributed data parallel (DDP) learning strategy
#[cfg(feature = "ddp")]
pub fn ddp<B: AutodiffBackend, LC: LearnerComponentTypes>(
    devices: Vec<LearnerDevice<LC>>,
    config: CollectiveConfig,
) -> LearningStrategy<LC> {
    LearningStrategy::DistributedDataParallel { devices, config }
}

impl<LC: LearnerComponentTypes> Default for LearningStrategy<LC> {
    fn default() -> Self {
        Self::SingleDevice(Default::default())
    }
}

/// Provides the `fit` function for any learning strategy
pub trait LearningMethod<LC: LearnerComponentTypes> {
    /// The dataloaders after being prepared for this trainin strategy
    ///
    /// (eg: splitting for multiple devices)
    type PreparedDataloaders;
    /// The model after being prepared for this training strategy
    ///
    /// The prepared model will be correctly initialized on the proper device for training.
    type PreparedModel;

    /// Fit the learner's model with this strategy.
    fn fit(
        &self,
        mut learner: Learner<LC>,
        dataloader_train: TrainLoader<LC>,
        dataloader_valid: ValidLoader<LC>,
    ) -> TrainingResult<LC::InnerModel> {
        let mut model = learner.model;
        let mut optim = learner.optim;
        let mut lr_scheduler = learner.lr_scheduler;
        let checkpoint = learner.checkpoint;

        let starting_epoch = match checkpoint {
            Some(checkpoint) => {
                if let Some(checkpointer) = &mut learner.checkpointer {
                    (model, optim, lr_scheduler) = checkpointer.load_checkpoint(
                        model,
                        optim,
                        lr_scheduler,
                        &Default::default(), // Load the checkpoint on the default device.
                        checkpoint,
                    );
                }
                checkpoint + 1
            }
            None => 1,
        };

        let dataloaders = self.prepare_dataloaders(dataloader_train, dataloader_valid);
        let model = self.prepare_model(model);

        // Training loop
        let mut components = LearnerComponents {
            optim,
            lr_scheduler,
            num_epochs: learner.num_epochs,
            checkpointer: learner.checkpointer,
            grad_accumulation: learner.grad_accumulation,
            interrupter: learner.interrupter,
            early_stopping: learner.early_stopping,
            event_processor: learner.event_processor,
            event_store: learner.event_store,
        };
        // Event processor start training
        components
            .event_processor
            .process_train(LearnerEvent::Start);
        let (model, mut event_processor) =
            self.learn(model, dataloaders, starting_epoch, components);

        let summary = learner.summary.and_then(|summary| {
            summary
                .init()
                .map(|summary| summary.with_model(model.to_string()))
                .ok()
        });

        // Signal training end. For the TUI renderer, this handles the exit & return to main screen.
        event_processor.process_train(LearnerEvent::End(summary));

        let model = model.valid();
        let renderer = event_processor.renderer();

        TrainingResult::<LC::InnerModel> { model, renderer }
    }

    /// Prepare the dataloaders for this strategy.
    /// The output will be used in [the learn function](Self::learn)
    fn prepare_dataloaders(
        &self,
        dataloader_train: TrainLoader<LC>,
        dataloader_valid: ValidLoader<LC>,
    ) -> Self::PreparedDataloaders;

    /// Prepare the model for this training strategy.
    /// The output will be used in [the learn function](Self::learn)
    fn prepare_model(&self, model: LC::Model) -> Self::PreparedModel;

    /// Training loop for this strategy
    fn learn(
        &self,
        model: Self::PreparedModel,
        dataloaders: Self::PreparedDataloaders,
        starting_epoch: usize,
        components: LearnerComponents<LC>,
    ) -> (LC::Model, LC::EventProcessor);
}

/// Struct to minimise parameters passed to [LearningMethod::learn]
/// These components are used during training
pub struct LearnerComponents<LC: LearnerComponentTypes> {
    /// The [Optimizer](LearnerComponentTypes::Optimizer) used for the training.
    pub optim: LC::Optimizer,
    /// The [learning rate scheduler](LearnerComponentTypes::LrScheduler) used for the training.
    pub lr_scheduler: LC::LrScheduler,
    /// The number of epochs the training should last.
    pub num_epochs: usize,
    /// Enables gradients accumulation.
    pub grad_accumulation: Option<usize>,
    /// A [LearnerCheckpointer](LearnerCheckpointer) used to save and load training checkpoints.
    pub checkpointer: Option<LearnerCheckpointer<LC>>,
    /// An [Interupter](Interrupter) that allows aborting the training/evaluation process early.
    pub interrupter: Interrupter,
    /// [Cloneable reference to an early stopping strategy](EarlyStoppingStrategyRef).
    pub early_stopping: Option<EarlyStoppingStrategyRef>,
    /// An [EventProcessor](LearnerComponentTypes::EventProcessor) that processes events happening during training and validation.
    pub event_processor: LC::EventProcessor,
    /// A reference to an [EventStoreClient](EventStoreClient).
    pub event_store: Arc<EventStoreClient>,
}
