use std::sync::Arc;

#[cfg(feature = "ddp")]
use burn_collective::CollectiveConfig;
use burn_core::{module::AutodiffModule, prelude::Backend};

use crate::{
    LearnerV2, MultiDeviceOptim, ParadigmComponents, SupervisedLearningComponents,
    TrainingComponents, TrainingResult,
    components_v2::{LearningComponents, TrainLoaderV2, ValidLoaderV2},
    metric::processor::{EventProcessorTraining, LearnerEvent},
};

type LearnerDevice<LC> = <<LC as LearningComponents>::Backend as Backend>::Device;

/// A reference to an implementation of SupervisedLearningStrategy.
pub type CustomLearningStrategyV2<SC> = Arc<dyn SupervisedLearningStrategy<SC>>;

/// How should the learner run the learning for the model
#[derive(Clone)]
pub enum TrainingStrategy<SC: SupervisedLearningComponents> {
    /// Training on one device
    SingleDevice(LearnerDevice<SC::LC>),
    /// Performs data-parallel distributed training where the optimization is
    /// done on an elected master device.
    MultiDevice(Vec<LearnerDevice<SC::LC>>, MultiDeviceOptim),
    /// Training using a custom learning strategy
    CustomSingleDevice(CustomLearningStrategyV2<SC>),
    /// Training with input distributed across devices, each device has its own copy of the model.
    /// Collective ops are used to sync the gradients after each pass.
    #[cfg(feature = "ddp")]
    DistributedDataParallel {
        /// Devices on this node for the DDP
        devices: Vec<LearnerDevice<SC::LC>>,

        /// The configuration for collective operations
        /// num_devices is ignored
        config: CollectiveConfig,
    },
}

/// Constructor for a distributed data parallel (DDP) learning strategy
#[cfg(feature = "ddp")]
pub fn ddp_v2<SC: SupervisedLearningComponents>(
    devices: Vec<LearnerDevice<SC::LC>>,
    config: CollectiveConfig,
) -> TrainingStrategy<SC> {
    TrainingStrategy::DistributedDataParallel { devices, config }
}

impl<SC: SupervisedLearningComponents> Default for TrainingStrategy<SC> {
    fn default() -> Self {
        Self::SingleDevice(Default::default())
    }
}

/// Provides the `fit` function for any learning strategy
pub trait SupervisedLearningStrategy<SC: SupervisedLearningComponents> {
    /// Train the learner's model with this strategy.
    fn train(
        &self,
        mut learner: LearnerV2<SC::LC>,
        dataloader_train: TrainLoaderV2<SC::LC, SC::LD>,
        dataloader_valid: ValidLoaderV2<SC::LC, SC::LD>,
        mut training_components: TrainingComponents<SC>,
    ) -> TrainingResult<SC::InnerModel> {
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
            .process_train(LearnerEvent::Start);
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
