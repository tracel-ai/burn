use super::{TrainStep, TrainerCallback, ValidStep};
use crate::data::dataloader::DataLoader;
use crate::train::{CheckpointModel, TrainerItem};
use std::sync::Arc;

pub trait Train<M, D> {
    fn train(self, model: M, data: D) -> M;
}

#[derive(new)]
pub struct TrainerData<T, V> {
    pub train: Arc<dyn DataLoader<T>>,
    pub valid: Arc<dyn DataLoader<V>>,
}

pub struct Trainer<TO, VO> {
    callback: Box<dyn TrainerCallback<TrainerItem<TO>, TrainerItem<VO>>>,
    num_epochs: usize,
    checkpoint: Option<usize>,
}

impl<TO, VO> Trainer<TO, VO> {
    pub fn new(
        callback: Box<dyn TrainerCallback<TrainerItem<TO>, TrainerItem<VO>>>,
        num_epochs: usize,
        checkpoint: Option<usize>,
    ) -> Self {
        Self {
            num_epochs,
            callback,
            checkpoint,
        }
    }
}

impl<M> Train<M, TrainerData<<M as TrainStep>::Input, <M as ValidStep>::Input>>
    for Trainer<<M as TrainStep>::Output, <M as ValidStep>::Output>
where
    M: TrainStep + ValidStep + CheckpointModel,
{
    fn train(
        mut self,
        mut learner: M,
        data: TrainerData<<M as TrainStep>::Input, <M as ValidStep>::Input>,
    ) -> M {
        let starting_epoch = match self.checkpoint {
            Some(checkpoint) => {
                learner.load_checkpoint(checkpoint);
                checkpoint
            }
            None => 1,
        };

        for epoch in starting_epoch..self.num_epochs + 1 {
            run_step(
                epoch,
                self.num_epochs,
                &data.train,
                &mut |item| TrainStep::step(&mut learner, item),
                &mut |log| self.callback.on_train_item(log),
            );
            self.callback.on_train_end_epoch();

            run_step(
                epoch,
                self.num_epochs,
                &data.valid,
                &mut |item| ValidStep::step(&learner, item),
                &mut |log| self.callback.on_valid_item(log),
            );
            self.callback.on_valid_end_epoch();
            learner.checkpoint(epoch);
        }

        learner
    }
}

pub fn run_step<I, O, F, FL>(
    epoch: usize,
    num_epochs: usize,
    dataloader: &Arc<dyn DataLoader<I>>,
    func: &mut F,
    func_log: &mut FL,
) where
    F: FnMut(I) -> O,
    FL: FnMut(TrainerItem<O>),
{
    let mut iterator = dataloader.iter();
    let mut iteration = 0;

    while let Some(item) = iterator.next() {
        let progress = iterator.progress();
        iteration += 1;

        let item = func(item);
        func_log(TrainerItem::new(
            item, progress, epoch, num_epochs, iteration,
        ));
    }
}
