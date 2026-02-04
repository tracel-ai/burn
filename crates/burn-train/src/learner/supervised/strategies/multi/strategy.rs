use crate::{
    Learner, LearningComponentsTypes, MultiDeviceOptim, SupervisedLearningStrategy,
    SupervisedTrainingEventProcessor, TrainLoader, TrainingBackend, TrainingComponents,
    TrainingModel, ValidLoader,
    multi::epoch::MultiDeviceTrainEpoch,
    single::{TrainingLoop, epoch::SingleDeviceValidEpoch},
};
use burn_core::{data::dataloader::split::split_dataloader, tensor::Device};

pub struct MultiDeviceLearningStrategy<LC: LearningComponentsTypes> {
    devices: Vec<Device<TrainingBackend<LC>>>,
    optim: MultiDeviceOptim,
}
impl<LC: LearningComponentsTypes> MultiDeviceLearningStrategy<LC> {
    pub fn new(devices: Vec<Device<TrainingBackend<LC>>>, optim: MultiDeviceOptim) -> Self {
        Self { devices, optim }
    }
}

impl<LC: LearningComponentsTypes> SupervisedLearningStrategy<LC>
    for MultiDeviceLearningStrategy<LC>
{
    fn fit(
        &self,
        training_components: TrainingComponents<LC>,
        mut learner: Learner<LC>,
        dataloader_train: TrainLoader<LC>,
        dataloader_valid: ValidLoader<LC>,
        starting_epoch: usize,
    ) -> (TrainingModel<LC>, SupervisedTrainingEventProcessor<LC>) {
        let main_device = self.devices.first().unwrap();

        // `MultiDevicesTrainStep` has one worker per device, so we use a fixed device strategy
        // for each (worker) data loader. This matches the expected device on the worker, so we
        // don't have to move the data between devices.
        let dataloader_train = split_dataloader(dataloader_train, &self.devices);
        let dataloader_valid = dataloader_valid.to_device(main_device);

        learner.fork(main_device);
        let mut event_processor = training_components.event_processor;
        let mut checkpointer = training_components.checkpointer;
        let mut early_stopping = training_components.early_stopping;

        let epoch_train = MultiDeviceTrainEpoch::<LC>::new(
            dataloader_train.clone(),
            training_components.grad_accumulation,
        );
        let epoch_valid: SingleDeviceValidEpoch<LC> =
            SingleDeviceValidEpoch::new(dataloader_valid.clone());

        for training_progress in TrainingLoop::new(starting_epoch, training_components.num_epochs) {
            let epoch = training_progress.items_processed;
            epoch_train.run(
                &mut learner,
                &training_progress,
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
                &training_progress,
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

        (learner.model(), event_processor)
    }
}
