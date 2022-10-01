use crate::data::dataloader::DataLoader;
use crate::tensor::backend::ADBackend;
use crate::train::{Learner, LearnerCheckpoint, Train, TrainerItem};
use std::sync::Arc;

pub trait SupervisedTrainerCallback<T, V>: Send {
    fn on_train_item(&mut self, _item: T) {}
    fn on_valid_item(&mut self, _item: V) {}
    fn on_train_end_epoch(&mut self) {}
    fn on_valid_end_epoch(&mut self) {}
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
    checkpoint: Option<usize>,
    _b: B,
}

impl<B, TO, VO> SupervisedTrainer<B, TO, VO>
where
    B: ADBackend,
{
    pub fn new(
        callback: Box<dyn SupervisedTrainerCallback<TrainerItem<TO>, TrainerItem<VO>>>,
        num_epochs: usize,
        checkpoint: Option<usize>,
    ) -> Self {
        Self {
            num_epochs,
            callback,
            checkpoint,
            _b: B::default(),
        }
    }
}

impl<B, TI, VI, L, TO, VO> Train<L, SupervisedData<TI, VI>> for SupervisedTrainer<B, TO, VO>
where
    B: ADBackend,
    L: Learner<TI, VI, TO, VO, Backend = B>,
    L: LearnerCheckpoint,
{
    fn train(mut self, mut learner: L, data: SupervisedData<TI, VI>) -> L {
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
