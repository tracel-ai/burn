use super::{Learner, TrainerItem};
use crate::data::dataloader::DataLoader;
use crate::tensor::backend::ADBackend;
use std::sync::Arc;

pub trait SupervisedTrainerCallback<T, V>: Send {
    fn on_train_item(&mut self, item: T);
    fn on_valid_item(&mut self, item: V);
    fn on_train_end_epoch(&mut self);
    fn on_valid_end_epoch(&mut self);
}

pub trait Train<M, D> {
    fn train(self, model: M, data: D) -> M;
}

#[derive(new)]
pub struct SupervisedData<T, V> {
    pub train: Arc<dyn DataLoader<T>>,
    pub valid: Arc<dyn DataLoader<V>>,
}

pub struct SupervisedTrainer<B, TO, VO>
where
    B: ADBackend,
{
    callback: Box<dyn SupervisedTrainerCallback<TrainerItem<TO>, TrainerItem<VO>>>,
    num_epochs: usize,
    _b: B,
}

impl<B, TO, VO> SupervisedTrainer<B, TO, VO>
where
    B: ADBackend,
{
    pub fn new(
        callback: Box<dyn SupervisedTrainerCallback<TrainerItem<TO>, TrainerItem<VO>>>,
        num_epochs: usize,
    ) -> Self {
        Self {
            num_epochs,
            callback,
            _b: B::default(),
        }
    }
}

impl<B, T, V, L, TO, VO> Train<L, SupervisedData<T, V>> for SupervisedTrainer<B, TO, VO>
where
    B: ADBackend,
    L: Learner<T, V, TO, VO, Backend = B>,
{
    fn train(mut self, mut learner: L, data: SupervisedData<T, V>) -> L {
        for epoch in 0..self.num_epochs {
            run_step(
                epoch,
                self.num_epochs,
                &data.train,
                &mut |item| learner.train(item),
                &mut |log| self.callback.on_train_item(log),
            );
            self.callback.on_train_end_epoch();

            run_step(
                epoch,
                self.num_epochs,
                &data.valid,
                &mut |item| learner.valid(item),
                &mut |log| self.callback.on_valid_item(log),
            );
            self.callback.on_valid_end_epoch();
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
