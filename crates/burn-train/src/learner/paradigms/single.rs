use crate::{
    LearnerComponents, LearnerV2, LearningMethod, ParadigmComponents, ParadigmInputTrain,
    ParadigmInputValid, ParadigmOutputTrain, ParadigmOutputValid, SupervisedComponents,
    TrainLoader, TrainStep, TrainingComponents, ValidLoader, ValidStep,
    components_v2::{LearnerComponentTypesV2, LearningDataV2, TrainLoaderV2, ValidLoaderV2},
    learner::{
        early_stopping,
        paradigms::{
            SingleDeviceTrainEpochV2, SingleDeviceValidEpochV2, SupervisedLearningStrategy,
        },
        strategies::single::epoch::{SingleDeviceTrainEpoch, SingleDeviceValidEpoch},
    },
};
use burn_core::{module::Module, tensor::Device};
use std::{marker::PhantomData, sync::Arc};

/// Simplest learning strategy possible, with only a single devices doing both the training and
/// validation.
pub struct SingleDeviceLearningStrategyV2<SC: SupervisedComponents> {
    device: Device<SC::Backend>,
}
impl<SC: SupervisedComponents> SingleDeviceLearningStrategyV2<SC> {
    pub fn new(device: Device<SC::Backend>) -> Self {
        Self { device }
    }
}

pub type CustomSingleDeviceLearningStrategy<LC> = Arc<
    dyn LearningMethod<
            LC,
            PreparedDataloaders = (TrainLoader<LC>, ValidLoader<LC>),
            PreparedModel = <LC as LearnerComponentTypesV2>::Model,
        >,
>;

impl<SC: SupervisedComponents> SupervisedLearningStrategy<SC>
    for SingleDeviceLearningStrategyV2<SC>
// where
//     SC: ParadigmComponents,
//     <SC::LearnerComponents as LearnerComponentTypesV2>::Model:
//         TrainStep<ParadigmInputTrain<SC>, ParadigmOutputTrain<SC>>,
//     <SC::LearnerComponents as LearnerComponentTypesV2>::InnerModel:
//         ValidStep<ParadigmInputValid<SC>, ParadigmOutputValid<SC>>,
{
    fn fit(
        &self,
        training_components: TrainingComponents<SC>,
        learner: LearnerV2<SC::LC>,
        dataloader_train: TrainLoaderV2<SC::LC, SC::LD>,
        dataloader_valid: ValidLoaderV2<SC::LC, SC::LD>,
        starting_epoch: usize,
    ) -> (SC::Model, <SC::PC as ParadigmComponents>::EventProcessor) {
        let dataloader_train = dataloader_train.to_device(&self.device);
        let dataloader_valid = dataloader_valid.to_device(&self.device);
        let mut learner = learner.fork(&self.device);
        let mut event_processor = training_components.event_processor;
        let mut checkpointer = training_components.checkpointer;
        let mut early_stopping = training_components.early_stopping;
        let num_epochs = training_components.num_epochs;

        for epoch in starting_epoch..training_components.num_epochs + 1 {
            let epoch_train: SingleDeviceTrainEpochV2<SC> = SingleDeviceTrainEpochV2::new(
                dataloader_train.clone(),
                epoch,
                num_epochs,
                training_components.grad_accumulation,
            );
            (learner, event_processor) =
                epoch_train.run(learner, event_processor, &training_components.interrupter);

            if training_components.interrupter.should_stop() {
                break;
            }

            let epoch_valid: SingleDeviceValidEpochV2<SC> =
                SingleDeviceValidEpochV2::new(dataloader_valid.clone(), epoch, num_epochs);
            event_processor =
                epoch_valid.run(&learner, event_processor, &training_components.interrupter);

            if let Some(checkpointer) = &mut checkpointer {
                checkpointer.checkpoint(
                    &learner.model,
                    &learner.optim,
                    &learner.lr_scheduler,
                    epoch,
                    &training_components.event_store,
                );
            }

            if let Some(early_stopping) = &mut early_stopping
                && early_stopping.should_stop(epoch, &training_components.event_store)
            {
                break;
            }
        }

        (learner.model, event_processor)
    }
}
