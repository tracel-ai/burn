use burn_core::data::dataloader::DataLoader;
use burn_core::tensor::backend::AutodiffBackend;
use burn_core::{lr_scheduler::LrScheduler, module::AutodiffModule, optim::GradientsAccumulator};
use std::sync::Arc;

use crate::components::OutputTrain;
use crate::metric::processor::{Event, EventProcessor, LearnerItem};
use crate::{TrainStep, ValidLoader, ValidStep};
use crate::{components::LearnerComponentTypes, learner::base::TrainingInterrupter};

/// A validation epoch.
#[derive(new)]
pub struct SingleDeviceValidEpoch<LC: LearnerComponentTypes> {
    dataloader: ValidLoader<LC>,
    epoch: usize,
    epoch_total: usize,
}

/// A training epoch.
#[derive(new)]
pub struct SingleDeviceTrainEpoch<B: AutodiffBackend, TI> {
    dataloader: Arc<dyn DataLoader<B, TI>>,
    epoch: usize,
    epoch_total: usize,
    grad_accumulation: Option<usize>,
}

impl<LC: LearnerComponentTypes> SingleDeviceValidEpoch<LC> {
    /// Runs the validation epoch.
    ///
    /// # Arguments
    ///
    /// * `model` - The model to validate.
    /// * `processor` - The event processor to use.
    pub fn run(
        &self,
        model: &LC::Model,
        processor: &mut LC::EventProcessor,
        interrupter: &TrainingInterrupter,
    ) {
        log::info!("Executing validation step for epoch {}", self.epoch);
        let model = model.valid();

        let mut iterator = self.dataloader.iter();
        let mut iteration = 0;

        while let Some(item) = iterator.next() {
            let progress = iterator.progress();
            iteration += 1;

            let item = model.step(item);
            let item = LearnerItem::new(
                item,
                progress,
                self.epoch,
                self.epoch_total,
                iteration,
                None,
            );

            processor.process_valid(Event::ProcessedItem(item));

            if interrupter.should_stop() {
                log::info!("Training interrupted.");
                break;
            }
        }
        processor.process_valid(Event::EndEpoch(self.epoch));
    }
}

impl<B: AutodiffBackend, TI> SingleDeviceTrainEpoch<B, TI> {
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
    pub fn run<LC: LearnerComponentTypes<Backend = B>>(
        &mut self,
        mut model: LC::Model,
        mut optim: LC::Optimizer,
        scheduler: &mut LC::LrScheduler,
        processor: &mut LC::EventProcessor,
        interrupter: &TrainingInterrupter,
    ) -> (LC::Model, LC::Optimizer)
    where
        LC::Model: TrainStep<TI, OutputTrain<LC>>,
    {
        log::info!("Executing training step for epoch {}", self.epoch,);

        // Single device / dataloader
        let mut iterator = self.dataloader.iter();
        let mut iteration = 0;
        let mut accumulator = GradientsAccumulator::new();
        let mut accumulation_current = 0;

        while let Some(item) = iterator.next() {
            iteration += 1;
            let lr = scheduler.step();
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
                self.epoch,
                self.epoch_total,
                iteration,
                Some(lr),
            );

            processor.process_train(Event::ProcessedItem(item));

            if interrupter.should_stop() {
                log::info!("Training interrupted.");
                break;
            }
        }
        processor.process_train(Event::EndEpoch(self.epoch));

        self.epoch += 1;

        (model, optim)
    }
}
