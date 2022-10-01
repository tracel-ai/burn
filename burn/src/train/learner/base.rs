use crate::data::dataloader::DataLoader;
use crate::module::ADModule;
use crate::optim::Optimizer;
use crate::tensor::backend::Backend;
use crate::train::checkpoint::Checkpointer;
use crate::train::{TrainerCallback, TrainerItem};
use burn_tensor::Gradients;
use std::sync::Arc;

#[derive(new)]
pub struct Learner<M, O, CM, CO, TO, VO> {
    model: M,
    optim: O,
    checkpointer_model: Option<CM>,
    checkpointer_optimizer: Option<CO>,
    callback: Box<dyn TrainerCallback<TrainerItem<TO>, TrainerItem<VO>>>,
    num_epochs: usize,
    checkpoint: Option<usize>,
}

#[derive(new)]
pub struct TrainOutput<TO> {
    grads: Gradients,
    item: TO,
}

pub trait TrainStep<TI, TO> {
    fn step(&self, item: TI) -> TrainOutput<TO>;
}

pub trait ValidStep<VI, VO> {
    fn step(&self, item: VI) -> VO;
}

impl<M, O, CM, CO, TO, VO> Learner<M, O, CM, CO, TO, VO>
where
    VO: Send + Sync + 'static,
    TO: Send + Sync + 'static,
    M: ADModule,
    O: Optimizer<Backend = M::Backend>,
    CM: Checkpointer<<M::Backend as Backend>::Elem>,
    CO: Checkpointer<<M::Backend as Backend>::Elem>,
{
    pub fn fit<TI, VI>(mut self, data: (Arc<dyn DataLoader<TI>>, Arc<dyn DataLoader<VI>>)) -> M
    where
        M: TrainStep<TI, TO>,
        M::InnerModule: ValidStep<VI, VO>,
    {
        let (dataloader_train, dataloader_valid) = data;

        let starting_epoch = match self.checkpoint {
            Some(checkpoint) => {
                self.load_checkpoint(checkpoint);
                checkpoint
            }
            None => 1,
        };

        for epoch in starting_epoch..self.num_epochs + 1 {
            let mut iterator = dataloader_train.iter();
            let mut iteration = 0;

            while let Some(item) = iterator.next() {
                let progress = iterator.progress();
                iteration += 1;

                let item = self.train_step(item);
                self.callback.on_train_item(TrainerItem::new(
                    item,
                    progress,
                    epoch,
                    self.num_epochs,
                    iteration,
                ));
            }
            self.callback.on_train_end_epoch();

            let mut iterator = dataloader_valid.iter();
            let mut iteration = 0;

            while let Some(item) = iterator.next() {
                let progress = iterator.progress();
                iteration += 1;

                let item = self.valid_step(item);
                self.callback.on_valid_item(TrainerItem::new(
                    item,
                    progress,
                    epoch,
                    self.num_epochs,
                    iteration,
                ));
            }
            self.callback.on_valid_end_epoch();
            self.checkpoint(epoch);
        }

        self.model
    }

    fn train_step<TI>(&mut self, item: TI) -> TO
    where
        M: TrainStep<TI, TO>,
    {
        let item = self.model.step(item);
        self.model.update_params(&item.grads, &mut self.optim);

        item.item
    }

    fn valid_step<VI>(&mut self, item: VI) -> VO
    where
        M::InnerModule: ValidStep<VI, VO>,
    {
        self.model.inner().step(item)
    }

    fn checkpoint(&self, epoch: usize) {
        if let Some(checkpointer) = &self.checkpointer_model {
            checkpointer.save(epoch, self.model.state()).unwrap();
        }
        if let Some(checkpointer) = &self.checkpointer_optimizer {
            checkpointer
                .save(epoch, self.optim.state(&self.model))
                .unwrap();
        }
    }

    fn load_checkpoint(&mut self, epoch: usize) {
        if let Some(checkpointer) = &self.checkpointer_model {
            let state = checkpointer.restore(epoch).unwrap();
            self.model.load(&state).unwrap();
        }

        if let Some(checkpointer) = &self.checkpointer_optimizer {
            let state = checkpointer.restore(epoch).unwrap();
            self.optim.load(&self.model, &state).unwrap();
        }
    }
}
