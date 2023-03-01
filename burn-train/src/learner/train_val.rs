use super::Learner;
use crate::train::MultiDevicesTrainStep;
use crate::LearnerItem;
use burn_core::data::dataloader::DataLoader;
use burn_core::module::ADModule;
use burn_core::optim::{
    convert_grads, to_device_grads, GradientsAccumulator, GradientsParams, Optimizer,
};
use burn_core::tensor::backend::ADBackend;
use std::sync::Arc;

pub struct TrainOutput<TO> {
    pub grads: GradientsParams,
    pub item: TO,
}

impl<TO> TrainOutput<TO> {
    pub fn new<M: ADModule>(
        module: &M,
        grads: <M::ADBackend as ADBackend>::Gradients,
        item: TO,
    ) -> Self {
        let grads = convert_grads(grads, module);

        Self { grads, item }
    }
}

pub trait TrainStep<TI, TO> {
    fn step(&self, item: TI) -> TrainOutput<TO>;
}

pub trait ValidStep<VI, VO> {
    fn step(&self, item: VI) -> VO;
}

impl<M, O, TO, VO> Learner<M, O, TO, VO>
where
    VO: Send + Sync + 'static,
    TO: Send + Sync + 'static,
    M: ADModule,
    O: Optimizer<Backend = M::Backend>,
{
    pub fn fit<TI, VI>(
        mut self,
        dataloader_train: Arc<dyn DataLoader<TI>>,
        dataloader_valid: Arc<dyn DataLoader<VI>>,
    ) -> M
    where
        TI: Send + 'static,
        TO: Send + 'static,
        M: TrainStep<TI, TO> + Send + Clone + 'static,
        M::InnerModule: ValidStep<VI, VO>,
    {
        log::info!("Fitting {}", self.model.to_string());

        let starting_epoch = match self.checkpoint {
            Some(checkpoint) => {
                self.load_checkpoint(checkpoint);
                checkpoint
            }
            None => 1,
        };

        // The reference model is always on the first device provided.
        if let Some(device) = self.devices.get(0) {
            self.model.to_device(device);
            self.model.detach();
        }

        for epoch in starting_epoch..self.num_epochs + 1 {
            if self.devices.len() > 1 {
                self.train_step_multi_devices(&dataloader_train, epoch);
            } else {
                self.train_step(&dataloader_train, epoch);
            }

            self.valid_step(&dataloader_valid, epoch);
            self.checkpoint(epoch);
        }

        self.model
    }

    fn train_step_multi_devices<TI>(
        &mut self,
        dataloader_train: &Arc<dyn DataLoader<TI>>,
        epoch: usize,
    ) where
        TI: Send + 'static,
        TO: Send + 'static,
        M: TrainStep<TI, TO> + Send + Clone + 'static,
    {
        log::info!(
            "Executing training step for epoch {} on devices {:?}",
            epoch,
            self.devices
        );

        let mut iterator = dataloader_train.iter();
        let mut iteration = 0;
        let mut accumulator = GradientsAccumulator::new();
        let mut accumulation_current = 0;

        let accumulation = self.grad_accumulation.unwrap_or(1) * self.devices.len();
        let step = MultiDevicesTrainStep::new(&self.devices);

        // The main device is always the first in the list.
        let device_main = self.devices.get(0).unwrap().clone();

        loop {
            let items = step.step(&mut iterator, &self.model);
            if items.is_empty() {
                break;
            }

            for mut item in items {
                iteration += 1;
                let progress = iterator.progress();

                to_device_grads(&mut item.grads, device_main.clone(), &self.model);
                log::info!("Updated device");
                accumulator.accumulate(&self.model, item.grads);
                accumulation_current += 1;

                if accumulation <= accumulation_current {
                    let grads = accumulator.grads();
                    self.optim.update_module(&mut self.model, grads);
                    accumulation_current = 0;
                }

                self.callback.on_train_item(LearnerItem::new(
                    item.item,
                    progress,
                    epoch,
                    self.num_epochs,
                    iteration,
                ));
            }
        }

        self.callback.on_train_end_epoch(epoch);
    }

    fn train_step<TI>(&mut self, dataloader_train: &Arc<dyn DataLoader<TI>>, epoch: usize)
    where
        TI: Send + 'static,
        TO: Send + 'static,
        M: TrainStep<TI, TO> + Send + Clone + 'static,
    {
        log::info!("Executing training step for epoch {}", epoch);

        let mut iterator = dataloader_train.iter();
        let mut iteration = 0;
        let mut accumulator = GradientsAccumulator::new();
        let mut accumulation_current = 0;

        while let Some(item) = iterator.next() {
            iteration += 1;

            let progress = iterator.progress();
            let item = self.model.step(item);

            match self.grad_accumulation {
                Some(accumulation) => {
                    accumulator.accumulate(&self.model, item.grads);
                    accumulation_current += 1;

                    if accumulation <= accumulation_current {
                        let grads = accumulator.grads();
                        self.optim.update_module(&mut self.model, grads);
                        accumulation_current = 0;
                    }
                }
                None => self.optim.update_module(&mut self.model, item.grads),
            }

            self.callback.on_train_item(LearnerItem::new(
                item.item,
                progress,
                epoch,
                self.num_epochs,
                iteration,
            ));
        }
        self.callback.on_train_end_epoch(epoch);
    }

    fn valid_step<VI>(&mut self, dataloader_valid: &Arc<dyn DataLoader<VI>>, epoch: usize)
    where
        M::InnerModule: ValidStep<VI, VO>,
    {
        log::info!("Executing validation step for epoch {}", epoch);

        let model = self.model.inner();

        let mut iterator = dataloader_valid.iter();
        let mut iteration = 0;

        while let Some(item) = iterator.next() {
            let progress = iterator.progress();
            iteration += 1;

            let item = model.step(item);
            self.callback.on_valid_item(LearnerItem::new(
                item,
                progress,
                epoch,
                self.num_epochs,
                iteration,
            ));
        }
        self.callback.on_valid_end_epoch(epoch);
    }
}
