use crate::{
    Learner, LearnerEvent, LearningComponentsTypes, MultiDeviceOptim, SupervisedLearningStrategy,
    SupervisedTrainingEventProcessor, TrainLoader, TrainingComponents, TrainingModel, ValidLoader,
    metric::processor::EventProcessorTraining,
    multi::epoch::MultiDeviceTrainEpoch,
    single::{TrainingLoop, epoch::SingleDeviceValidEpoch},
};
use burn_core::{data::dataloader::split::split_dataloader, tensor::Device};

pub struct MultiDeviceLearningStrategy {
    devices: Vec<Device>,
    optim: MultiDeviceOptim,
}
impl MultiDeviceLearningStrategy {
    pub fn new(devices: Vec<Device>, optim: MultiDeviceOptim) -> Self {
        Self { devices, optim }
    }
}

impl<LC: LearningComponentsTypes> SupervisedLearningStrategy<LC> for MultiDeviceLearningStrategy {
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
        let train_total_items = dataloader_train.num_items();
        let dataloader_train = split_dataloader(dataloader_train, &self.devices);
        let dataloader_valid = dataloader_valid.to_device(&main_device.clone().inner());
        let valid_total_items = dataloader_valid.num_items();

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

            event_processor.process_train(LearnerEvent::StartSplit(train_total_items));
            epoch_train.run(
                &mut learner,
                &training_progress,
                &mut event_processor,
                &training_components.interrupter,
                self.devices.to_vec(),
                self.optim,
            );
            event_processor.process_train(LearnerEvent::EndSplit(epoch));

            if training_components.interrupter.should_stop() {
                let reason = training_components
                    .interrupter
                    .get_message()
                    .unwrap_or(String::from("Reason unknown"));
                log::info!("Training interrupted: {reason}");
                break;
            }

            // After OptimSharded training, model parameters are scattered across
            // devices. Fork back to main_device before single-device validation.
            if matches!(self.optim, MultiDeviceOptim::OptimSharded) {
                learner.fork(main_device);
            }

            event_processor.process_valid(LearnerEvent::StartSplit(valid_total_items));
            epoch_valid.run(
                &learner,
                &training_progress,
                &mut event_processor,
                &training_components.interrupter,
            );
            event_processor.process_valid(LearnerEvent::EndSplit(epoch));
            event_processor.process_train(LearnerEvent::EndEpoch(epoch));

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
