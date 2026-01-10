use crate::{
    Learner, ParadigmComponentsTypes, SupervisedLearningComponentsTypes,
    SupervisedLearningStrategy, TrainBackend, TrainingComponents,
    components::{TrainLoader, ValidLoader},
    single::epoch::{SingleDeviceTrainEpoch, SingleDeviceValidEpoch},
};
use burn_core::tensor::Device;

/// Simplest learning strategy possible, with only a single devices doing both the training and
/// validation.
pub struct SingleDevicetrainingStrategy<SC: SupervisedLearningComponentsTypes> {
    device: Device<TrainBackend<SC::LC>>,
}
impl<SC: SupervisedLearningComponentsTypes> SingleDevicetrainingStrategy<SC> {
    pub fn new(device: Device<TrainBackend<SC::LC>>) -> Self {
        Self { device }
    }
}

impl<SC: SupervisedLearningComponentsTypes> SupervisedLearningStrategy<SC>
    for SingleDevicetrainingStrategy<SC>
{
    fn fit(
        &self,
        training_components: TrainingComponents<SC>,
        learner: Learner<SC::LC>,
        dataloader_train: TrainLoader<SC::LC, SC::LD>,
        dataloader_valid: ValidLoader<SC::LC, SC::LD>,
        starting_epoch: usize,
    ) -> (
        SC::Model,
        <SC::PC as ParadigmComponentsTypes>::EventProcessor,
    ) {
        let dataloader_train = dataloader_train.to_device(&self.device);
        let dataloader_valid = dataloader_valid.to_device(&self.device);
        let mut learner = learner.fork(&self.device);
        let mut event_processor = training_components.event_processor;
        let mut checkpointer = training_components.checkpointer;
        let mut early_stopping = training_components.early_stopping;
        let num_epochs = training_components.num_epochs;

        let epoch_train: SingleDeviceTrainEpoch<SC> = SingleDeviceTrainEpoch::new(
            dataloader_train,
            num_epochs,
            training_components.grad_accumulation,
        );
        let epoch_valid: SingleDeviceValidEpoch<SC> =
            SingleDeviceValidEpoch::new(dataloader_valid.clone(), num_epochs);

        for epoch in starting_epoch..training_components.num_epochs + 1 {
            epoch_train.run(
                &mut learner,
                epoch,
                &mut event_processor,
                &training_components.interrupter,
            );

            if training_components.interrupter.should_stop() {
                let reason = training_components
                    .interrupter
                    .get_message()
                    .unwrap_or(String::from("Reason unknown"));
                log::info!("Training interrupted: {reason}");
                break;
            }

            epoch_valid.run(
                &learner,
                epoch,
                &mut event_processor,
                &training_components.interrupter,
            );

            if let Some(checkpointer) = &mut checkpointer {
                checkpointer.checkpoint(&learner, epoch, &training_components.event_store);
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
