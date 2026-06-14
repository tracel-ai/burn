use crate::{
    EarlyStoppingStrategyRef, InferenceModel, Interrupter, Learner, LearnerSummaryConfig,
    LearningCheckpointer, LearningResult, SupervisedTrainingEventProcessor, TrainLoader,
    TrainingModel, ValidLoader,
    components::LearningComponentsTypes,
    metric::{
        processor::{EventProcessorTraining, LearnerEvent},
        store::EventStoreClient,
    },
};
use burn_core::tensor::distributed::{DistributedConfig, DistributedContext};
use burn_core::{module::AutodiffModule, prelude::Device};
use std::sync::Arc;

/// A reference to an implementation of SupervisedLearningStrategy.
pub type CustomLearningStrategy<LC> = Arc<dyn SupervisedLearningStrategy<LC>>;

#[derive(Clone, Copy, Debug)]
/// Determine how the optimization is performed when training with multiple devices.
pub enum MultiDeviceOptim {
    /// The optimization is done on an elected device.
    OptimMainDevice,
    /// The optimization is sharded across all devices.
    OptimSharded,
}

/// Describes where training runs.
pub enum ExecutionStrategy {
    /// Training on one device
    SingleDevice(Device),
    /// Performs data-parallel distributed training where the optimization is
    /// done on an elected master device.
    MultiDevice(Vec<Device>, MultiDeviceOptim),
    /// Training with input distributed across devices, each device has its own copy of the model.
    /// Collective ops are used to sync the gradients after each pass.
    DistributedDataParallel {
        /// Devices on this node for the DDP
        devices: Vec<Device>,
        /// The distributed runtime.
        context: DistributedContext,
    },
}

impl ExecutionStrategy {
    /// Returns the primary device responsible for coordination.
    pub fn main_device(&self) -> &Device {
        match self {
            ExecutionStrategy::SingleDevice(device) => device,
            ExecutionStrategy::MultiDevice(devices, _optim) => &devices[0],
            ExecutionStrategy::DistributedDataParallel {
                devices,
                context: _,
            } => &devices[0],
        }
    }

    /// Creates a strategy for a single device.
    pub fn single(device: Device) -> Self {
        Self::SingleDevice(device)
    }

    /// Creates a multi-device strategy.
    pub fn multi(devices: Vec<Device>, optim: MultiDeviceOptim) -> Self {
        Self::MultiDevice(devices, optim)
    }
}

impl ExecutionStrategy {
    /// Creates a distributed data parallel (DDP) strategy.
    pub fn ddp(devices: Vec<Device>, config: DistributedConfig) -> Self {
        let context = DistributedContext::init(devices.clone(), config);
        Self::DistributedDataParallel { devices, context }
    }
}

/// How should the learner run the learning for the model
pub enum TrainingStrategy<LC: LearningComponentsTypes> {
    /// Default training loop with specified device strategy.
    Default(ExecutionStrategy),
    /// Training using a custom learning strategy
    Custom(CustomLearningStrategy<LC>),
}

impl<LC: LearningComponentsTypes> From<ExecutionStrategy> for TrainingStrategy<LC> {
    fn from(value: ExecutionStrategy) -> Self {
        Self::Default(value)
    }
}

impl<LC: LearningComponentsTypes> Default for TrainingStrategy<LC> {
    fn default() -> Self {
        Self::Default(ExecutionStrategy::SingleDevice(Default::default()))
    }
}

/// Struct to minimise parameters passed to [SupervisedLearningStrategy::train].
/// These components are used during training.
pub struct TrainingComponents<LC: LearningComponentsTypes> {
    /// The total number of epochs
    pub num_epochs: usize,
    /// The epoch number from which to continue the training.
    pub checkpoint: Option<usize>,
    /// A checkpointer used to load and save learner checkpoints.
    pub checkpointer: Option<LearningCheckpointer<LC>>,
    /// Enables gradients accumulation.
    pub grad_accumulation: Option<usize>,
    /// An [Interupter](Interrupter) that allows aborting the training/evaluation process early.
    pub interrupter: Interrupter,
    /// Cloneable reference to an early stopping strategy.
    pub early_stopping: Option<EarlyStoppingStrategyRef>,
    /// An [EventProcessor](crate::EventProcessorTraining) that processes events happening during training and validation.
    pub event_processor: SupervisedTrainingEventProcessor<LC>,
    /// A reference to an [EventStoreClient](EventStoreClient).
    pub event_store: Arc<EventStoreClient>,
    /// Config for creating a summary of the learning
    pub summary: Option<LearnerSummaryConfig>,
}

/// Provides the `fit` function for any learning strategy
pub trait SupervisedLearningStrategy<LC: LearningComponentsTypes> {
    /// Train the learner's model with this strategy.
    fn train(
        &self,
        mut learner: Learner<LC>,
        dataloader_train: TrainLoader<LC>,
        dataloader_valid: ValidLoader<LC>,
        mut training_components: TrainingComponents<LC>,
    ) -> LearningResult<InferenceModel<LC>> {
        let starting_epoch = match training_components.checkpoint {
            Some(checkpoint) => {
                if let Some(checkpointer) = &mut training_components.checkpointer {
                    learner =
                        checkpointer.load_checkpoint(learner, &Default::default(), checkpoint);
                }
                checkpoint + 1
            }
            None => 1,
        };

        let summary_config = training_components.summary.clone();

        // Event processor start training
        training_components
            .event_processor
            .process_train(LearnerEvent::Start {
                total_epochs: training_components.num_epochs,
            });
        // Training loop
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

        LearningResult::<InferenceModel<LC>> { model, renderer }
    }

    /// Training loop for this strategy
    fn fit(
        &self,
        training_components: TrainingComponents<LC>,
        learner: Learner<LC>,
        dataloader_train: TrainLoader<LC>,
        dataloader_valid: ValidLoader<LC>,
        starting_epoch: usize,
    ) -> (TrainingModel<LC>, SupervisedTrainingEventProcessor<LC>);
}
