use crate::learner::base::Interrupter;
use crate::learner::train_val::TrainStep;
use crate::metric::processor::{EventProcessorTraining, LearnerEvent, LearnerItem};
use crate::train::MultiDevicesTrainStep;
use crate::{
    Learner, LearningComponentsTypes, MultiDeviceOptim, ParadigmComponentsTypes,
    SupervisedLearningComponentsTypes, TrainBackend, TrainLoader,
};
use burn_core::prelude::DeviceOps;
use burn_core::tensor::Device;
use burn_core::tensor::backend::DeviceId;
use burn_optim::MultiGradientsParams;
use burn_optim::{GradientsAccumulator, lr_scheduler::LrScheduler};
use std::collections::HashMap;

/// A training epoch.
#[derive(new)]
pub struct MultiDeviceTrainEpoch<SC: SupervisedLearningComponentsTypes> {
    dataloaders: Vec<TrainLoader<SC::LC, SC::LD>>,
    epoch_total: usize,
    grad_accumulation: Option<usize>,
}

impl<SC: SupervisedLearningComponentsTypes> MultiDeviceTrainEpoch<SC> {
    /// Runs the training epoch on multiple devices.
    ///
    /// # Arguments
    ///
    /// * `model` - The model to train.
    /// * `optim` - The optimizer to use.
    /// * `lr_scheduler` - The learning rate scheduler to use.
    /// * `processor` - The event processor to use.
    /// * `devices` - The devices to use.
    ///
    /// # Returns
    ///
    /// The trained model and the optimizer.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        learner: &mut Learner<SC::LC>,
        epoch: usize,
        event_processor: &mut <SC::PC as ParadigmComponentsTypes>::EventProcessor,
        interrupter: &Interrupter,
        devices: Vec<Device<TrainBackend<SC::LC>>>,
        strategy: MultiDeviceOptim,
    ) {
        match strategy {
            MultiDeviceOptim::OptimMainDevice => {
                self.run_optim_main(learner, epoch, event_processor, interrupter, devices)
            }
            MultiDeviceOptim::OptimSharded => {
                self.run_optim_distr(learner, epoch, event_processor, interrupter, devices)
            }
        }
    }

    fn run_optim_main(
        &self,
        learner: &mut Learner<SC::LC>,
        epoch: usize,
        event_processor: &mut <SC::PC as ParadigmComponentsTypes>::EventProcessor,
        interrupter: &Interrupter,
        devices: Vec<Device<TrainBackend<SC::LC>>>,
    ) {
        log::info!(
            "Executing training step for epoch {} on devices {:?}",
            epoch,
            devices
        );

        let mut iterators = self
            .dataloaders
            .iter()
            .map(|d| d.iter())
            .collect::<Vec<_>>();
        let mut iteration = 0;
        let mut accumulator = GradientsAccumulator::new();
        let mut accumulation_current = 0;

        let accumulation = self.grad_accumulation.unwrap_or(1);
        let step = MultiDevicesTrainStep::<SC>::new(&devices);

        // The main device is always the first in the list.
        let device_main = devices.first().expect("A minimum of one device.").clone();

        let mut model = learner.model.clone();
        let mut optim = learner.optim.clone();
        let mut lr_scheduler = learner.lr_scheduler.clone();

        loop {
            let (items, progress) = step.step(iterators.as_mut_slice(), &model);
            if items.is_empty() {
                break;
            }

            let lr = lr_scheduler.step();

            let mut progress_items = Vec::with_capacity(items.len());
            for item in items.into_iter() {
                let grads = item.output.grads.to_device(&device_main, &model);
                accumulator.accumulate(&model, grads);
                progress_items.push(item.output.item);
            }

            accumulation_current += 1;

            if accumulation <= accumulation_current {
                let grads = accumulator.grads();
                model = model.optimize(&mut optim, lr, grads);
                accumulation_current = 0;
            }

            for item in progress_items {
                iteration += 1;
                let item = LearnerItem::new(
                    item,
                    progress.clone(),
                    epoch,
                    self.epoch_total,
                    iteration,
                    Some(lr),
                );

                event_processor.process_train(LearnerEvent::ProcessedItem(item));
            }

            if interrupter.should_stop() {
                log::info!("Training interrupted.");
                break;
            }
        }

        learner.model = model;
        learner.optim = optim;
        learner.lr_scheduler = lr_scheduler;

        event_processor.process_train(LearnerEvent::EndEpoch(epoch));
    }

    fn run_optim_distr(
        &self,
        learner: &mut Learner<SC::LC>,
        epoch: usize,
        event_processor: &mut <SC::PC as ParadigmComponentsTypes>::EventProcessor,
        interrupter: &Interrupter,
        devices: Vec<Device<TrainBackend<SC::LC>>>,
    ) {
        log::info!(
            "Executing training step for epoch {} on devices {:?}",
            epoch,
            devices
        );

        let mut iterators = self
            .dataloaders
            .iter()
            .map(|d| d.iter())
            .collect::<Vec<_>>();
        let mut iteration = 0;
        let mut accumulators = HashMap::<
            DeviceId,
            GradientsAccumulator<<SC::LC as LearningComponentsTypes>::Model>,
        >::new();
        for device in devices.iter() {
            accumulators.insert(device.to_id(), GradientsAccumulator::new());
        }
        let mut accumulation_current = 0;

        let accumulation = self.grad_accumulation.unwrap_or(1);
        let step = MultiDevicesTrainStep::<SC>::new(&devices);

        let mut model = learner.model.clone();
        let mut optim = learner.optim.clone();
        let mut lr_scheduler = learner.lr_scheduler.clone();

        loop {
            let (items, progress) = step.step(iterators.as_mut_slice(), &model);
            if items.is_empty() {
                break;
            }

            let lr = lr_scheduler.step();

            let mut progress_items = Vec::with_capacity(items.len());
            for item in items.into_iter() {
                let accumulator = accumulators.get_mut(&item.device).unwrap();
                accumulator.accumulate(&model, item.output.grads);
                progress_items.push(item.output.item);
            }

            accumulation_current += 1;

            if accumulation <= accumulation_current {
                let mut grads = MultiGradientsParams::default();
                for (device_id, accumulator) in accumulators.iter_mut() {
                    let grad = accumulator.grads();
                    grads.grads.push((grad, *device_id));
                }
                model = model.optimize_multi(&mut optim, lr, grads);
                accumulation_current = 0;
            }

            for item in progress_items {
                iteration += 1;
                let item = LearnerItem::new(
                    item,
                    progress.clone(),
                    epoch,
                    self.epoch_total,
                    iteration,
                    Some(lr),
                );

                event_processor.process_train(LearnerEvent::ProcessedItem(item));
            }

            if interrupter.should_stop() {
                log::info!("Training interrupted.");
                break;
            }
        }

        learner.model = model;
        learner.optim = optim;
        learner.lr_scheduler = lr_scheduler;

        event_processor.process_train(LearnerEvent::EndEpoch(epoch));
    }
}
