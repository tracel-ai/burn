use burn_tensor::backend::ADBackend;

use super::Learner;
use crate::data::dataloader::DataLoader;
use crate::module::ADModule;
use crate::optim::Optimizer;
use crate::train::LearnerItem;
use std::sync::Arc;

#[derive(new)]
pub struct TrainOutput<TO, G> {
    grads: G,
    item: TO,
}

pub trait TrainStep<TI, TO, G> {
    fn step(&self, item: TI) -> TrainOutput<TO, G>;
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
        M: TrainStep<TI, TO, <M::ADBackend as ADBackend>::Gradients>,
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

        for epoch in starting_epoch..self.num_epochs + 1 {
            self.train_step(&dataloader_train, epoch);
            self.valid_step(&dataloader_valid, epoch);
            self.checkpoint(epoch);
        }

        self.model
    }

    fn train_step<TI>(&mut self, dataloader_train: &Arc<dyn DataLoader<TI>>, epoch: usize)
    where
        M: TrainStep<TI, TO, <M::ADBackend as ADBackend>::Gradients>,
    {
        log::info!("Executing training step for epoch {}", epoch);

        let mut iterator = dataloader_train.iter();
        let mut iteration = 0;

        while let Some(item) = iterator.next() {
            let progress = iterator.progress();
            iteration += 1;

            let item = self.model.step(item);
            self.model.update_params(&item.grads, &mut self.optim);

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
