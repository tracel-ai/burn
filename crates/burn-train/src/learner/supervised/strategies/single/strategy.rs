use crate::{
    EventProcessorTraining, Learner, LearnerEvent, LearningComponentsTypes,
    SupervisedLearningStrategy, SupervisedTrainingEventProcessor, TrainLoader, TrainingComponents,
    TrainingModel, ValidLoader,
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

        let progress = Progress {
            items_processed: self.next_iteration,
            items_total: self.total_iteration,
            unit: Some("epochs".to_string()),
        };

        self.next_iteration += 1;
        Some(progress)
    }
}

impl<LC: LearningComponentsTypes> SupervisedLearningStrategy<LC> for SingleDeviceTrainingStrategy {
    fn fit(
        &self,
        training_components: TrainingComponents<LC>,
        mut learner: Learner<LC>,
        dataloader_train: TrainLoader<LC>,
        dataloader_valid: ValidLoader<LC>,
        starting_epoch: usize,
    ) -> (TrainingModel<LC>, SupervisedTrainingEventProcessor<LC>) {
        let dataloader_train = dataloader_train.to_device(&self.device);
        let train_total_items = dataloader_train.num_items();
        let dataloader_valid = dataloader_valid.to_device(&self.device.clone().inner());
        let valid_total_items = dataloader_valid.num_items();
        learner.fork(&self.device);
        let mut event_processor = training_components.event_processor;
        let mut checkpointer = training_components.checkpointer;
        let mut early_stopping = training_components.early_stopping;

        let epoch_train: SingleDeviceTrainEpoch<LC> =
            SingleDeviceTrainEpoch::new(dataloader_train, training_components.grad_accumulation);
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
