use crate::metric::processor::{EventProcessorTraining, LearnerEvent, LearnerItem};
use crate::{MultiDevicesTrainStep, TrainLoader, TrainStep};
use crate::{components::LearnerComponentTypes, learner::base::Interrupter};
use burn_core::tensor::backend::Backend;
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
        let mut accumulator = GradientsAccumulator::new();
        let mut accumulation_current = 0;

        let accumulation = self.grad_accumulation.unwrap_or(1) * devices.len();
        let step = MultiDevicesTrainStep::<LC>::new(&devices);

        // The main device is always the first in the list.
        let device_main = devices.first().expect("A minimum of one device.").clone();
        let mut interrupted = false;

        loop {
            let (items, progress) = step.step(iterators.as_mut_slice(), &model);
            if items.is_empty() {
                break;
            }

            for item in items {
                iteration += 1;
                let lr = lr_scheduler.step();

                let grads = item.grads.to_device(&device_main, &model);

                accumulator.accumulate(&model, grads);
                accumulation_current += 1;

                if accumulation <= accumulation_current {
                    let grads = accumulator.grads();
                    model = model.optimize(&mut optim, lr, grads);
                    accumulation_current = 0;
                }

                let item = LearnerItem::new(
                    item.item,
                    progress.clone(),
                    self.epoch,
                    self.epoch_total,
                    iteration,
                    Some(lr),
                );

                processor.process_train(LearnerEvent::ProcessedItem(item));

                if interrupter.should_stop() {
                    log::info!("Training interrupted.");
                    interrupted = true;
                    break;
                }
            }

            if interrupted {
                break;
            }
        }

        processor.process_train(LearnerEvent::EndEpoch(self.epoch));

        self.epoch += 1;

        (model, optim)
    }
}
