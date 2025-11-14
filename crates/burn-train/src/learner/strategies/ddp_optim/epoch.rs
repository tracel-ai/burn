use std::collections::HashMap;

use crate::metric::processor::{EventProcessorTraining, LearnerEvent, LearnerItem};
use crate::{MultiDevicesTrainStep, TrainLoader, TrainStep};
use crate::{components::LearnerComponentTypes, learner::base::Interrupter};
use burn_core::prelude::DeviceOps;
use burn_core::tensor::backend::{Backend, DeviceId};
use burn_optim::MultiGradientsParams;
use burn_optim::{GradientsAccumulator, lr_scheduler::LrScheduler};

/// A training epoch.
#[derive(new)]
pub struct MultiDeviceTrainEpoch<LC: LearnerComponentTypes> {
    dataloaders: Vec<TrainLoader<LC>>,
    epoch: usize,
    epoch_total: usize,
    grad_accumulation: Option<usize>,
}

impl<LC: LearnerComponentTypes> MultiDeviceTrainEpoch<LC> {
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
    pub fn run(
        &mut self,
        mut model: LC::Model,
        mut optim: LC::Optimizer,
        lr_scheduler: &mut LC::LrScheduler,
        processor: &mut LC::EventProcessor,
        devices: Vec<<LC::Backend as Backend>::Device>,
        interrupter: &Interrupter,
    ) -> (LC::Model, LC::Optimizer) {
        log::info!(
            "Executing training step for epoch {} on devices {:?}",
            self.epoch,
            devices
        );

        let mut iterators = self
            .dataloaders
            .iter()
            .map(|d| d.iter())
            .collect::<Vec<_>>();
        let mut iteration = 0;
        let mut accumulators = HashMap::<DeviceId, GradientsAccumulator<LC::Model>>::new();
        for device in devices.iter() {
            accumulators.insert(device.to_id(), GradientsAccumulator::new());
        }
        let mut accumulation_current = 0;

        let accumulation = self.grad_accumulation.unwrap_or(1);
        let step = MultiDevicesTrainStep::<LC>::new(&devices);

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
                iteration += devices.len();
                let item = LearnerItem::new(
                    item,
                    progress.clone(),
                    self.epoch,
                    self.epoch_total,
                    iteration,
                    Some(lr),
                );

                processor.process_train(LearnerEvent::ProcessedItem(item));
            }

            if interrupter.should_stop() {
                log::info!("Training interrupted.");
                break;
            }
        }

        processor.process_train(LearnerEvent::EndEpoch(self.epoch));

        self.epoch += 1;

        (model, optim)
    }
}
