use crate::{
    EventProcessorTraining, Learner, LearnerEvent, LearnerModel, SupervisedLearningStrategy,
    SupervisedTrainingEventProcessor, TrainLoader, TrainingComponents, ValidLoader,
    single::epoch::{SingleDeviceTrainEpoch, SingleDeviceValidEpoch},
};
use burn_core::{data::dataloader::Progress, tensor::Device};

/// Simplest learning strategy possible, with only a single devices doing both the training and
/// validation.
pub struct SingleDeviceTrainingStrategy {
    device: Device,
}
impl SingleDeviceTrainingStrategy {
    pub fn new(device: Device) -> Self {
        Self { device }
    }
}

#[derive(new)]
pub(crate) struct TrainingLoop {
    next_iteration: usize,
    total_iteration: usize,
}

/// An iterator that returns the progress of the training.
impl Iterator for TrainingLoop {
    type Item = Progress;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_iteration > self.total_iteration {
            return None;
        }

        let progress = Progress::new(
            self.next_iteration,
            self.total_iteration,
            Some("epochs".to_string()),
        );

        self.next_iteration += 1;
        Some(progress)
    }
}

impl<M: LearnerModel> SupervisedLearningStrategy<M> for SingleDeviceTrainingStrategy {
    fn fit(
        &self,
        training_components: TrainingComponents<M>,
        mut learner: Learner<M>,
        dataloader_train: TrainLoader<M>,
        dataloader_valid: ValidLoader<M>,
        starting_epoch: usize,
    ) -> (M, SupervisedTrainingEventProcessor<M>) {
        let dataloader_train = dataloader_train.to_device(&self.device);
        let dataloader_valid = dataloader_valid.to_device(&self.device.clone().inner());
        learner.fork(&self.device);
        let mut event_processor = training_components.event_processor;
        let mut checkpointer = training_components.checkpointer;
        let mut early_stopping = training_components.early_stopping;

        let epoch_train: SingleDeviceTrainEpoch<M> =
            SingleDeviceTrainEpoch::new(dataloader_train, training_components.grad_accumulation);
        let epoch_valid: SingleDeviceValidEpoch<M> =
            SingleDeviceValidEpoch::new(dataloader_valid.clone());

        for training_progress in TrainingLoop::new(starting_epoch, training_components.num_epochs) {
            let epoch = training_progress.items_processed;

            epoch_train.run(
                &mut learner,
                &training_progress,
                &mut event_processor,
                &training_components.interrupter,
            );

            if training_components.interrupter.should_stop() {
                if let Some(interruption) = training_components.interrupter.interruption() {
                    let reason = interruption.reason.as_deref().unwrap_or("reason unknown");
                    log::info!("Training interrupted: {reason}");
                }
                break;
            }

            epoch_valid.run(
                &learner,
                &training_progress,
                &mut event_processor,
                &training_components.interrupter,
            );
            if training_components.interrupter.should_stop() {
                break;
            }
            training_components
                .interrupter
                .fail_on_error(event_processor.process_train(LearnerEvent::EndEpoch(epoch)));
            if checkpointer.is_some() || early_stopping.is_some() {
                training_components
                    .interrupter
                    .fail_on_error(event_processor.flush());
            }

            if training_components.interrupter.should_stop() {
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
