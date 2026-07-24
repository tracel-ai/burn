use crate::learner::base::Interrupter;
use crate::metric::processor::{EventProcessorTraining, LearnerEvent, TrainingItem};
use crate::{
    InferenceStep, Learner, LearnerModel, SupervisedTrainingEventProcessor, TrainLoader,
    ValidLoader,
};
use burn_core::data::dataloader::Progress;
use burn_optim::GradientsAccumulator;

/// A validation epoch.
#[derive(new)]
pub struct SingleDeviceValidEpoch<M: LearnerModel> {
    dataloader: ValidLoader<M>,
}

/// A training epoch.
#[derive(new)]
pub struct SingleDeviceTrainEpoch<M: LearnerModel> {
    dataloader: TrainLoader<M>,
    grad_accumulation: Option<usize>,
}

impl<M: LearnerModel> SingleDeviceValidEpoch<M> {
    /// Runs the validation epoch.
    ///
    /// # Arguments
    ///
    /// * `model` - The model to validate.
    /// * `processor` - The event processor to use.
    pub fn run(
        &self,
        learner: &Learner<M>,
        global_progress: &Progress,
        processor: &mut SupervisedTrainingEventProcessor<M>,
        interrupter: &Interrupter,
    ) {
        let epoch = global_progress.items_processed;
        log::info!("Executing validation step for epoch {}", epoch);
        let model = learner.model().valid();

        let mut iterator = self.dataloader.iter();
        let mut iteration = 0;

        while let Some(item) = iterator.next() {
            let item = match item {
                Ok(item) => item,
                Err(err) => {
                    interrupter.stop(Some(&format!("dataset error during validation: {err}")));
                    break;
                }
            };
            let progress = iterator.progress();
            iteration += 1;

            let item = InferenceStep::step(&model, item);
            let item = TrainingItem::new(item, progress, Some(iteration), None);

            processor.process_valid(LearnerEvent::ProcessedItem(item));

            if interrupter.should_stop() {
                break;
            }
        }
    }
}

impl<M: LearnerModel> SingleDeviceTrainEpoch<M> {
    /// Runs the training epoch.
    ///
    /// # Arguments
    ///
    /// * `model` - The model to train.
    /// * `optim` - The optimizer to use.
    /// * `scheduler` - The learning rate scheduler to use.
    /// * `processor` - The event processor to use.
    ///
    /// # Returns
    ///
    /// The trained model and the optimizer.
    pub fn run(
        &self,
        learner: &mut Learner<M>,
        global_progress: &Progress,
        processor: &mut SupervisedTrainingEventProcessor<M>,
        interrupter: &Interrupter,
    ) {
        let epoch = global_progress.items_processed;
        log::info!("Executing training step for epoch {}", epoch,);

        // Single device / dataloader
        let mut iterator = self.dataloader.iter();
        let mut iteration = 0;
        let mut accumulator = GradientsAccumulator::new();
        let mut accumulation_current = 0;

        while let Some(item) = iterator.next() {
            let item = match item {
                Ok(item) => item,
                Err(err) => {
                    interrupter.stop(Some(&format!("dataset error during training: {err}")));
                    break;
                }
            };
            iteration += 1;
            learner.lr_step();
            log::info!("Iteration {iteration}");

            let progress = iterator.progress();
            let item = learner.train_step(item);

            match self.grad_accumulation {
                Some(accumulation) => {
                    accumulator.accumulate(&learner.model(), item.grads);
                    accumulation_current += 1;

                    if accumulation <= accumulation_current {
                        let grads = accumulator.grads();

                        learner.optimizer_step(grads);
                        accumulation_current = 0;
                    }
                }
                None => learner.optimizer_step(item.grads),
            }

            let item = TrainingItem::new(
                item.item,
                progress,
                Some(iteration),
                Some(learner.lr_current()),
            );

            processor.process_train(LearnerEvent::ProcessedItem(item));

            if interrupter.should_stop() {
                break;
            }
        }
    }
}
