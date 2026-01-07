use crate::{
    Learner, MultiDeviceOptim, ParadigmComponentsTypes, SupervisedLearningComponentsTypes,
    SupervisedLearningStrategy, TrainBackend, TrainingComponents,
    components::{TrainLoader, ValidLoader},
    multi::epoch::MultiDeviceTrainEpoch,
    single::epoch::SingleDeviceValidEpoch,
};
use burn_core::{data::dataloader::split::split_dataloader, tensor::Device};

pub struct MultiDeviceLearningStrategy<SC: SupervisedLearningComponentsTypes> {
    devices: Vec<Device<TrainBackend<SC::LC>>>,
    optim: MultiDeviceOptim,
}
impl<SC: SupervisedLearningComponentsTypes> MultiDeviceLearningStrategy<SC> {
    pub fn new(devices: Vec<Device<TrainBackend<SC::LC>>>, optim: MultiDeviceOptim) -> Self {
        Self { devices, optim }
    }
}

impl<SC: SupervisedLearningComponentsTypes> SupervisedLearningStrategy<SC>
    for MultiDeviceLearningStrategy<SC>
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
        let main_device = self.devices.first().unwrap();

        // `MultiDevicesTrainStep` has one worker per device, so we use a fixed device strategy
        // for each (worker) data loader. This matches the expected device on the worker, so we
        // don't have to move the data between devices.
        let dataloader_train = split_dataloader(dataloader_train, &self.devices);
        let dataloader_valid = dataloader_valid.to_device(main_device);

        let mut learner = learner.fork(main_device);
        let mut event_processor = training_components.event_processor;
        let mut checkpointer = training_components.checkpointer;
        let mut early_stopping = training_components.early_stopping;
        let num_epochs = training_components.num_epochs;

        let epoch_train: MultiDeviceTrainEpoch<SC> = MultiDeviceTrainEpoch::new(
            dataloader_train.clone(),
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
                self.devices.to_vec(),
                self.optim,
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
