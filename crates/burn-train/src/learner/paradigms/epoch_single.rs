use burn_core::data::dataloader::DataLoader;
use burn_core::module::AutodiffModule;
use burn_core::prelude::Backend;
use burn_core::tensor::backend::AutodiffBackend;
use burn_optim::{GradientsAccumulator, lr_scheduler::LrScheduler};
use std::sync::Arc;

use crate::components::OutputTrain;
use crate::components_v2::{
    LearnerComponentTypesV2, OutputTrainV2, OutputValidV2, TrainLoaderV2, ValidLoaderV2,
};
use crate::metric::processor::{EventProcessorTraining, LearnerEvent, LearnerItem};
use crate::{
    LearnerV2, ParadigmComponents, ParadigmInputTrain, ParadigmInputValid, ParadigmOutputTrain,
    ParadigmOutputValid, SupervisedComponents, TrainStep, ValidLoader, ValidStep, learner,
};
use crate::{components::LearnerComponentTypes, learner::base::Interrupter};

/// A validation epoch.
#[derive(new)]
pub struct SingleDeviceValidEpochV2<SC: SupervisedComponents> {
    dataloader: ValidLoaderV2<SC::LC, SC::LD>,
    epoch: usize,
    epoch_total: usize,
}

/// A training epoch.
#[derive(new)]
pub struct SingleDeviceTrainEpochV2<SC: SupervisedComponents> {
    dataloader: TrainLoaderV2<SC::LC, SC::LD>,
    epoch: usize,
    epoch_total: usize,
    grad_accumulation: Option<usize>,
}

impl<SC: SupervisedComponents> SingleDeviceValidEpochV2<SC> {
    /// Runs the validation epoch.
    ///
    /// # Arguments
    ///
    /// * `model` - The model to validate.
    /// * `processor` - The event processor to use.
    pub fn run(
        self,
        learner: &LearnerV2<SC::LC>,
        mut processor: <SC::PC as ParadigmComponents>::EventProcessor,
        interrupter: &Interrupter,
    ) -> <SC::PC as ParadigmComponents>::EventProcessor {
        log::info!("Executing validation step for epoch {}", self.epoch);
        let model = learner.model.valid();

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

            processor.process_valid(LearnerEvent::ProcessedItem(item));

            if interrupter.should_stop() {
                log::info!("Training interrupted.");
                break;
            }
        }
        processor.process_valid(LearnerEvent::EndEpoch(self.epoch));
        processor
    }
}

impl<SC: SupervisedComponents> SingleDeviceTrainEpochV2<SC> {
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
        self,
        learner: LearnerV2<SC::LC>,
        mut processor: <SC::PC as ParadigmComponents>::EventProcessor,
        interrupter: &Interrupter,
    ) -> (
        LearnerV2<SC::LC>,
        <SC::PC as ParadigmComponents>::EventProcessor,
    ) {
        log::info!("Executing training step for epoch {}", self.epoch,);

        // Single device / dataloader
        let mut iterator = self.dataloader.iter();
        let mut iteration = 0;
        let mut accumulator = GradientsAccumulator::new();
        let mut accumulation_current = 0;
        let mut model = learner.model;
        let mut optim = learner.optim;
        let mut lr_scheduler = learner.lr_scheduler;

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
                self.epoch,
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
        processor.process_train(LearnerEvent::EndEpoch(self.epoch));

        (
            LearnerV2 {
                model,
                optim,
                lr_scheduler,
            },
            processor,
        )
    }
}
