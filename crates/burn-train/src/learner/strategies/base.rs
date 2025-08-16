use std::sync::Arc;

#[cfg(feature = "ddp")]
use burn_collective::CollectiveConfig;
use burn_core::tensor::backend::AutodiffBackend;

use crate::{
    EarlyStoppingStrategyRef, Learner, LearnerCheckpointer, TrainLoader, TrainingInterrupter,
    ValidLoader,
    components::LearnerComponentTypes,
    metric::{
        processor::{Event, EventProcessor},
        store::EventStoreClient,
    },
};

/// How should the learner run the learning for the model
#[derive(Clone)]
pub enum LearningStrategy<B: AutodiffBackend> {
    /// Training on one device
    SingleDevice(B::Device),

    /// Legacy implementation of local multi-device training
    MultiDeviceNaive(Vec<B::Device>),

    /// Training with input distributed across devices, each device has its own copy of the model.
    /// Collective ops are used to sync the gradients after each pass.
    #[cfg(feature = "ddp")]
    DistributedDataParallel {
        /// Devices on this node for the DDP
        devices: Vec<B::Device>,

        /// The configuration for collective operations
        /// num_devices is ignored
        config: CollectiveConfig,
    },
}

/// Constructor for a distributed data parallel (DDP) learning strategy
#[cfg(feature = "ddp")]
pub fn ddp<B: AutodiffBackend>(
    devices: Vec<B::Device>,
    config: CollectiveConfig,
) -> LearningStrategy<B> {
    LearningStrategy::DistributedDataParallel { devices, config }
}

impl<B: AutodiffBackend> Default for LearningStrategy<B> {
    fn default() -> Self {
        Self::SingleDevice(Default::default())
    }
}

/// Provides the `fit` function for any learning strategy
pub(crate) trait LearningMethod<LC: LearnerComponentTypes> {
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
    ) -> LC::Model {
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
        let components = LearnerComponents {
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
        let (model, mut event_processor) =
            self.learn(model, dataloaders, starting_epoch, components);

        // Signal training end. For the TUI renderer, this handles the exit & return to main screen.
        event_processor.process_train(Event::End);

        let summary = learner.summary;
        if let Some(summary) = summary {
            match summary.init() {
                Ok(summary) => {
                    println!("{}", summary.with_model(model.to_string()))
                }
                Err(err) => log::error!("Could not retrieve learner summary:\n{err}"),
            }
        }

        model
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
pub(crate) struct LearnerComponents<LC: LearnerComponentTypes> {
    pub optim: LC::Optimizer,
    pub lr_scheduler: LC::LrScheduler,
    pub num_epochs: usize,
    pub grad_accumulation: Option<usize>,
    pub checkpointer: Option<LearnerCheckpointer<LC>>,
    pub interrupter: TrainingInterrupter,
    pub early_stopping: Option<EarlyStoppingStrategyRef>,
    pub event_processor: LC::EventProcessor,
    pub event_store: Arc<EventStoreClient>,
}
