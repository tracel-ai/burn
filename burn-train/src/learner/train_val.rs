use super::Learner;

use crate::{TrainEpoch, ValidEpoch};
use burn_core::data::dataloader::DataLoader;
use burn_core::module::ADModule;
use burn_core::optim::{GradientsParams, Optimizer};
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
        let grads = GradientsParams::from_grads(grads, module);

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
                self = self.load_checkpoint(checkpoint);
                checkpoint
            }
            None => 1,
        };

        let mut model = self.model;
        let mut optim = self.optim;

        // The reference model is always on the first device provided.
        if let Some(device) = self.devices.get(0) {
            model = model.to_device(device).detach();
        }

        for epoch in starting_epoch..self.num_epochs + 1 {
            let epoch_train = TrainEpoch::new(
                dataloader_train.clone(),
                epoch,
                self.num_epochs,
                self.grad_accumulation,
            );

            if self.devices.is_empty() {
                (model, optim) = epoch_train.run(model, optim, &mut self.callback);
            } else {
                (model, optim) = epoch_train.run_multi_device(
                    model,
                    optim,
                    &mut self.callback,
                    self.devices.clone(),
                );
            }

            let epoch_valid = ValidEpoch::new(dataloader_valid.clone(), epoch, self.num_epochs);
            model = epoch_valid.run(model, &mut self.callback);

            Self::checkpoint(
                &model,
                &optim,
                &self.checkpointer_model,
                &self.checkpointer_optimizer,
                epoch,
            );
        }

        model
    }
}
