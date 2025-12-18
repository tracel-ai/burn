use crate::components::{TrainLoader, ValidLoader};
use crate::learner::base::Interrupter;
use crate::metric::processor::{EventProcessorTraining, LearnerEvent, LearnerItem};
use crate::{
    Learner, ParadigmComponentsTypes, SupervisedLearningComponentsTypes, TrainStep, ValidStep,
};
use burn_core::module::AutodiffModule;
use burn_optim::{GradientsAccumulator, lr_scheduler::LrScheduler};

/// A validation epoch.
#[derive(new)]
pub struct SingleDeviceValidEpoch<SC: SupervisedLearningComponentsTypes> {
    dataloader: ValidLoader<SC::LC, SC::LD>,
    epoch_total: usize,
}

/// A training epoch.
#[derive(new)]
pub struct SingleDeviceTrainEpoch<SC: SupervisedLearningComponentsTypes> {
    dataloader: TrainLoader<SC::LC, SC::LD>,
    epoch_total: usize,
    grad_accumulation: Option<usize>,
}

impl<SC: SupervisedLearningComponentsTypes> SingleDeviceValidEpoch<SC> {
    /// Runs the validation epoch.
    ///
    /// # Arguments
    ///
    /// * `model` - The model to validate.
    /// * `processor` - The event processor to use.
    pub fn run(
        &self,
        learner: &Learner<SC::LC>,
        epoch: usize,
        processor: &mut <SC::PC as ParadigmComponentsTypes>::EventProcessor,
        interrupter: &Interrupter,
    ) {
        log::info!("Executing validation step for epoch {}", epoch);
        let model = learner.model.valid();

        let mut iterator = self.dataloader.iter();
        let mut iteration = 0;

        while let Some(item) = iterator.next() {
            let progress = iterator.progress();
            iteration += 1;

            let item = model.step(item);
            let item = LearnerItem::new(item, progress, epoch, self.epoch_total, iteration, None);

            processor.process_valid(LearnerEvent::ProcessedItem(item));

            if interrupter.should_stop() {
                log::info!("Training interrupted.");
                break;
            }
        }
        processor.process_valid(LearnerEvent::EndEpoch(epoch));
    }
}

impl<SC: SupervisedLearningComponentsTypes> SingleDeviceTrainEpoch<SC> {
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
        learner: &mut Learner<SC::LC>,
        epoch: usize,
        processor: &mut <SC::PC as ParadigmComponentsTypes>::EventProcessor,
        interrupter: &Interrupter,
    ) {
        log::info!("Executing training step for epoch {}", epoch,);

        // Single device / dataloader
        let mut iterator = self.dataloader.iter();
        let mut iteration = 0;
        let mut accumulator = GradientsAccumulator::new();
        let mut accumulation_current = 0;

        let mut model = learner.model.clone();
        let mut optim = learner.optim.clone();
        let mut lr_scheduler = learner.lr_scheduler.clone();

        while let Some(item) = iterator.next() {
            iteration += 1;
            let lr = lr_scheduler.step();
            log::info!("Iteration {iteration}");

            let progress = iterator.progress();
            let item = model.step(item);

            match self.grad_accumulation {
                Some(accumulation) => {
                    accumulator.accumulate(&model, item.grads);
                    accumulation_current += 1;

                    if accumulation <= accumulation_current {
                        let grads = accumulator.grads();

                        model = model.optimize(&mut optim, lr, grads);
                        accumulation_current = 0;
                    }
                }
                None => model = model.optimize(&mut optim, lr, item.grads),
            }

            let item = LearnerItem::new(
                item.item,
                progress,
                epoch,
                self.epoch_total,
                iteration,
                Some(lr),
            );

            processor.process_train(LearnerEvent::ProcessedItem(item));

            if interrupter.should_stop() {
                log::info!("Training interrupted.");
                break;
            }
        }
        processor.process_train(LearnerEvent::EndEpoch(epoch));

        learner.model = model;
        learner.optim = optim;
        learner.lr_scheduler = lr_scheduler;
    }
}
