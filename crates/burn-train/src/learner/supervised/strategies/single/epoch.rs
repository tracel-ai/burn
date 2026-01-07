use crate::learner::base::Interrupter;
use crate::metric::processor::{EventProcessorTraining, LearnerEvent, LearnerItem};
use crate::{
    InferenceStep, Learner, LearningComponentsTypes, SupervisedTrainingEventProcessor, TrainLoader,
    ValidLoader,
};
use burn_core::module::AutodiffModule;
use burn_optim::GradientsAccumulator;

/// A validation epoch.
#[derive(new)]
pub struct SingleDeviceValidEpoch<LC: LearningComponentsTypes> {
    dataloader: ValidLoader<LC>,
    epoch_total: usize,
}

/// A training epoch.
#[derive(new)]
pub struct SingleDeviceTrainEpoch<LC: LearningComponentsTypes> {
    dataloader: TrainLoader<LC>,
    epoch_total: usize,
    grad_accumulation: Option<usize>,
}

impl<LC: LearningComponentsTypes> SingleDeviceValidEpoch<LC> {
    /// Runs the validation epoch.
    ///
    /// # Arguments
    ///
    /// * `model` - The model to validate.
    /// * `processor` - The event processor to use.
    pub fn run(
        &self,
        learner: &Learner<LC>,
        epoch: usize,
        processor: &mut SupervisedTrainingEventProcessor<LC>,
        interrupter: &Interrupter,
    ) {
        log::info!("Executing validation step for epoch {}", epoch);
        let model = learner.model().valid();

        let mut iterator = self.dataloader.iter();
        let mut iteration = 0;

        while let Some(item) = iterator.next() {
            let progress = iterator.progress();
            iteration += 1;

            let item = model.step(item);
            let item = LearnerItem::new(item, progress, epoch, self.epoch_total, iteration, None);

            processor.process_valid(LearnerEvent::ProcessedItem(item));

            if interrupter.should_stop() {
                break;
            }
        }
        processor.process_valid(LearnerEvent::EndEpoch(epoch));
    }
}

impl<LC: LearningComponentsTypes> SingleDeviceTrainEpoch<LC> {
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
        learner: &mut Learner<LC>,
        epoch: usize,
        processor: &mut SupervisedTrainingEventProcessor<LC>,
        interrupter: &Interrupter,
    ) {
        log::info!("Executing training step for epoch {}", epoch,);

        // Single device / dataloader
        let mut iterator = self.dataloader.iter();
        let mut iteration = 0;
        let mut accumulator = GradientsAccumulator::new();
        let mut accumulation_current = 0;

        while let Some(item) = iterator.next() {
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

            let item = LearnerItem::new(
                item.item,
                progress,
                epoch,
                self.epoch_total,
                iteration,
                Some(learner.lr_current()),
            );

            processor.process_train(LearnerEvent::ProcessedItem(item));

            if interrupter.should_stop() {
                break;
            }
        }
        processor.process_train(LearnerEvent::EndEpoch(epoch));
    }
}
