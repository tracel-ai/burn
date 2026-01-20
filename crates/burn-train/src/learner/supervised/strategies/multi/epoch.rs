use crate::learner::base::Interrupter;
use crate::metric::processor::{EventProcessorTraining, LearnerEvent, TrainingItem};
use crate::train::MultiDevicesTrainStep;
use crate::{
    Learner, LearningComponentsTypes, MultiDeviceOptim, SupervisedTrainingEventProcessor,
    TrainLoader, TrainingBackend,
};
use burn_core::data::dataloader::Progress;
use burn_core::prelude::DeviceOps;
use burn_core::tensor::Device;
use burn_core::tensor::backend::DeviceId;
use burn_optim::GradientsAccumulator;
use burn_optim::MultiGradientsParams;
use std::collections::HashMap;

/// A training epoch.
#[derive(new)]
pub struct MultiDeviceTrainEpoch<LC: LearningComponentsTypes> {
    dataloaders: Vec<TrainLoader<LC>>,
    grad_accumulation: Option<usize>,
}

impl<LC: LearningComponentsTypes> MultiDeviceTrainEpoch<LC> {
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
        learner: &mut Learner<LC>,
        global_progress: &Progress,
        event_processor: &mut SupervisedTrainingEventProcessor<LC>,
        interrupter: &Interrupter,
        devices: Vec<Device<TrainingBackend<LC>>>,
        strategy: MultiDeviceOptim,
    ) {
        match strategy {
            MultiDeviceOptim::OptimMainDevice => self.run_optim_main(
                learner,
                global_progress,
                event_processor,
                interrupter,
                devices,
            ),
            MultiDeviceOptim::OptimSharded => self.run_optim_distr(
                learner,
                global_progress,
                event_processor,
                interrupter,
                devices,
            ),
        }
    }

    fn run_optim_main(
        &self,
        learner: &mut Learner<LC>,
        global_progress: &Progress,
        event_processor: &mut SupervisedTrainingEventProcessor<LC>,
        interrupter: &Interrupter,
        devices: Vec<Device<TrainingBackend<LC>>>,
    ) {
        let epoch = global_progress.items_processed;
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
        let step = MultiDevicesTrainStep::<LC>::new(&devices);

        // The main device is always the first in the list.
        let device_main = devices.first().expect("A minimum of one device.").clone();

        loop {
            let (items, progress) = step.step(iterators.as_mut_slice(), &learner.model());
            if items.is_empty() {
                break;
            }

            learner.lr_step();

            let mut progress_items = Vec::with_capacity(items.len());
            for item in items.into_iter() {
                let grads = item.output.grads.to_device(&device_main, &learner.model());
                accumulator.accumulate(&learner.model(), grads);
                progress_items.push(item.output.item);
            }

            accumulation_current += 1;

            if accumulation <= accumulation_current {
                let grads = accumulator.grads();
                learner.optimizer_step(grads);
                accumulation_current = 0;
            }

            for item in progress_items {
                iteration += 1;
                let item = TrainingItem::new(
                    item,
                    progress.clone(),
                    global_progress.clone(),
                    Some(iteration),
                    Some(learner.lr_current()),
                );

                event_processor.process_train(LearnerEvent::ProcessedItem(item));
            }

            if interrupter.should_stop() {
                break;
            }
        }

        event_processor.process_train(LearnerEvent::EndEpoch(epoch));
    }

    fn run_optim_distr(
        &self,
        learner: &mut Learner<LC>,
        global_progress: &Progress,
        event_processor: &mut SupervisedTrainingEventProcessor<LC>,
        interrupter: &Interrupter,
        devices: Vec<Device<TrainingBackend<LC>>>,
    ) {
        let epoch = global_progress.items_processed;
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
            GradientsAccumulator<<LC as LearningComponentsTypes>::TrainingModel>,
        >::new();
        for device in devices.iter() {
            accumulators.insert(device.to_id(), GradientsAccumulator::new());
        }
        let mut accumulation_current = 0;

        let accumulation = self.grad_accumulation.unwrap_or(1);
        let step = MultiDevicesTrainStep::<LC>::new(&devices);

        loop {
            let (items, progress) = step.step(iterators.as_mut_slice(), &learner.model());
            if items.is_empty() {
                break;
            }

            learner.lr_step();

            let mut progress_items = Vec::with_capacity(items.len());
            for item in items.into_iter() {
                let accumulator = accumulators.get_mut(&item.device).unwrap();
                accumulator.accumulate(&learner.model(), item.output.grads);
                progress_items.push(item.output.item);
            }

            accumulation_current += 1;

            if accumulation <= accumulation_current {
                let mut grads = MultiGradientsParams::default();
                for (device_id, accumulator) in accumulators.iter_mut() {
                    let grad = accumulator.grads();
                    grads.grads.push((grad, *device_id));
                }
                learner.optimizer_step_multi(grads);
                accumulation_current = 0;
            }

            for item in progress_items {
                iteration += 1;
                let item = TrainingItem::new(
                    item,
                    progress.clone(),
                    global_progress.clone(),
                    Some(iteration),
                    Some(learner.lr_current()),
                );

                event_processor.process_train(LearnerEvent::ProcessedItem(item));
            }

            if interrupter.should_stop() {
                break;
            }
        }

        event_processor.process_train(LearnerEvent::EndEpoch(epoch));
    }
}
