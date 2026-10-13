use crate::{
    Learner, LearnerEvent, LearnerModel, MultiDeviceOptim, SupervisedLearningStrategy,
    SupervisedTrainingEventProcessor, TrainLoader, TrainingComponents, ValidLoader,
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

impl<M: LearnerModel> SupervisedLearningStrategy<M> for MultiDeviceLearningStrategy {
    fn fit(
        &self,
        training_components: TrainingComponents<M>,
        mut learner: Learner<M>,
        dataloader_train: TrainLoader<M>,
        dataloader_valid: ValidLoader<M>,
        starting_epoch: usize,
    ) -> (M, SupervisedTrainingEventProcessor<M>) {
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
        let interrupter = training_components.interrupter.clone();

        let epoch_train = MultiDeviceTrainEpoch::<M>::new(
            dataloader_train.clone(),
            training_components.grad_accumulation,
        );
        let epoch_valid: SingleDeviceValidEpoch<M> =
            SingleDeviceValidEpoch::new(dataloader_valid.clone());

        for training_progress in TrainingLoop::new(starting_epoch, training_components.num_epochs) {
            let epoch = training_progress.items_processed;

            interrupter.fail_on_error(event_processor.process_train(LearnerEvent::StartSplit {
                epoch_number: epoch,
                total_items: train_total_items,
            }));
            epoch_train.run(
                &mut learner,
                &training_progress,
                &mut event_processor,
                &interrupter,
                self.devices.to_vec(),
                self.optim,
            );
            interrupter.fail_on_error(event_processor.process_train(LearnerEvent::EndSplit(epoch)));

            if interrupter.should_stop() {
                if let Some(interruption) = interrupter.interruption() {
                    let reason = interruption.reason.as_deref().unwrap_or("reason unknown");
                    log::info!("Training interrupted: {reason}");
                }
                break;
            }

            // After OptimSharded training, model parameters are scattered across
            // devices. Fork back to main_device before single-device validation.
            if matches!(self.optim, MultiDeviceOptim::OptimSharded) {
                learner.fork(main_device);
            }

            interrupter.fail_on_error(event_processor.process_valid(LearnerEvent::StartSplit {
                epoch_number: epoch,
                total_items: valid_total_items,
            }));
            epoch_valid.run(
                &learner,
                &training_progress,
                &mut event_processor,
                &interrupter,
            );
            interrupter.fail_on_error(event_processor.process_valid(LearnerEvent::EndSplit(epoch)));
            interrupter.fail_on_error(event_processor.process_train(LearnerEvent::EndEpoch(epoch)));
            if checkpointer.is_some() || early_stopping.is_some() {
                interrupter.fail_on_error(event_processor.flush());
            }

            if interrupter.should_stop() {
                break;
            }

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
