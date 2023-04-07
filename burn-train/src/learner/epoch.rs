use burn_core::{
    data::dataloader::DataLoader,
    module::ADModule,
    optim::{GradientsAccumulator, Optimizer},
    tensor::backend::ADBackend,
};
use std::sync::Arc;

use crate::{LearnerCallback, LearnerItem, MultiDevicesTrainStep, TrainStep, ValidStep};

#[derive(new)]
pub struct ValidEpoch<VI> {
    dataloader: Arc<dyn DataLoader<VI>>,
    epoch: usize,
    epoch_total: usize,
}

#[derive(new)]
pub struct TrainEpoch<TI> {
    dataloader: Arc<dyn DataLoader<TI>>,
    epoch: usize,
    epoch_total: usize,
    grad_accumulation: Option<usize>,
}

impl<I> ValidEpoch<I> {
    pub fn run<B, M, TO, VO>(&self, model: &M, callback: &mut Box<dyn LearnerCallback<TO, VO>>)
    where
        B: ADBackend,
        M: ADModule<B>,
        M::InnerModule: ValidStep<I, VO>,
    {
        log::info!("Executing validation step for epoch {}", self.epoch);
        let model = model.valid();

        let mut iterator = self.dataloader.iter();
        let mut iteration = 0;

        while let Some(item) = iterator.next() {
            let progress = iterator.progress();
            iteration += 1;

            let item = model.step(item);
            callback.on_valid_item(LearnerItem::new(
                item,
                progress,
                self.epoch,
                self.epoch_total,
                iteration,
            ));
        }
        callback.on_valid_end_epoch(self.epoch);
    }
}

impl<TI> TrainEpoch<TI> {
    pub fn run<B, M, O, TO, VO>(
        &self,
        mut model: M,
        mut optim: O,
        callback: &mut Box<dyn LearnerCallback<TO, VO>>,
    ) -> (M, O)
    where
        B: ADBackend,
        M: ADModule<B>,
        O: Optimizer<M, B>,
        M: TrainStep<TI, TO>,
    {
        log::info!("Executing training step for epoch {}", self.epoch,);

        let mut iterator = self.dataloader.iter();
        let mut iteration = 0;
        let mut accumulator = GradientsAccumulator::new();
        let mut accumulation_current = 0;

        while let Some(item) = iterator.next() {
            iteration += 1;
            log::info!("Iteration {}", iteration);

            let progress = iterator.progress();
            let item = model.step(item);

            match self.grad_accumulation {
                Some(accumulation) => {
                    accumulator.accumulate(&model, item.grads);
                    accumulation_current += 1;

                    if accumulation <= accumulation_current {
                        let grads = accumulator.grads();
                        model = optim.step(model, grads);
                        accumulation_current = 0;
                    }
                }
                None => model = optim.step(model, item.grads),
            }

            callback.on_train_item(LearnerItem::new(
                item.item,
                progress,
                self.epoch,
                self.epoch_total,
                iteration,
            ));
        }
        callback.on_train_end_epoch(self.epoch);

        (model, optim)
    }
}

impl<TI> TrainEpoch<TI> {
    pub fn run_multi_device<B, M, O, TO, VO>(
        &self,
        mut model: M,
        mut optim: O,
        callback: &mut Box<dyn LearnerCallback<TO, VO>>,
        devices: Vec<B::Device>,
    ) -> (M, O)
    where
        B: ADBackend,
        M: ADModule<B> + 'static,
        O: Optimizer<M, B>,
        M: TrainStep<TI, TO>,
        TI: Send + 'static,
        TO: Send + 'static,
    {
        log::info!(
            "Executing training step for epoch {} on devices {:?}",
            self.epoch,
            devices
        );

        let mut iterator = self.dataloader.iter();
        let mut iteration = 0;
        let mut accumulator = GradientsAccumulator::new();
        let mut accumulation_current = 0;

        let accumulation = self.grad_accumulation.unwrap_or(1) * devices.len();
        let step = MultiDevicesTrainStep::new(&devices);

        // The main device is always the first in the list.
        let device_main = devices.get(0).unwrap().clone();

        loop {
            let items = step.step(&mut iterator, &model);
            if items.is_empty() {
                break;
            }

            for item in items {
                iteration += 1;
                let progress = iterator.progress();

                let grads = item.grads.to_device(&device_main, &model);

                accumulator.accumulate(&model, grads);
                accumulation_current += 1;

                if accumulation <= accumulation_current {
                    let grads = accumulator.grads();
                    model = optim.step(model, grads);
                    accumulation_current = 0;
                }

                callback.on_train_item(LearnerItem::new(
                    item.item,
                    progress,
                    self.epoch,
                    self.epoch_total,
                    iteration,
                ));
            }
        }

        callback.on_train_end_epoch(self.epoch);

        (model, optim)
    }
}
