use super::{Learner, TrainerItem};
use crate::data::dataloader::DataLoader;
use crate::data::dataloader::Detach;
use crate::tensor::backend::ADBackend;
use crate::train::logger::Logger;
use std::sync::Arc;

pub struct SupervisedTrainer<B, T, V, L, TO, VO>
where
    B: ADBackend,
    L: Learner<T, V, TO, VO, Backend = B>,
{
    dataloader_train: Arc<dyn DataLoader<T>>,
    dataloader_valid: Arc<dyn DataLoader<V>>,
    logger_train: Box<dyn Logger<TrainerItem<TO>>>,
    logger_valid: Box<dyn Logger<TrainerItem<VO>>>,
    learner: L,
    _b: B,
}

impl<B, T, V, L, TO, VO> SupervisedTrainer<B, T, V, L, TO, VO>
where
    B: ADBackend,
    T: Detach,
    L: Learner<T, V, TO, VO, Backend = B>,
{
    pub fn new(
        dataloader_train: Arc<dyn DataLoader<T>>,
        dataloader_valid: Arc<dyn DataLoader<V>>,
        logger_train: Box<dyn Logger<TrainerItem<TO>>>,
        logger_valid: Box<dyn Logger<TrainerItem<VO>>>,
        learner: L,
    ) -> Self {
        Self {
            dataloader_train,
            dataloader_valid,
            learner,
            logger_train,
            logger_valid,
            _b: B::default(),
        }
    }

    pub fn run(mut self, num_epochs: usize) -> L {
        for epoch in 0..num_epochs {
            run_step(
                epoch,
                num_epochs,
                &self.dataloader_train,
                &mut self.logger_train,
                &mut |item| self.learner.train(item.detach()),
            );

            run_step(
                epoch,
                num_epochs,
                &self.dataloader_valid,
                &mut self.logger_valid,
                &mut |item| self.learner.valid(item),
            );
        }

        self.learner
    }
}

pub fn run_step<I, O, F>(
    epoch: usize,
    num_epochs: usize,
    dataloader: &Arc<dyn DataLoader<I>>,
    logger: &mut Box<dyn Logger<TrainerItem<O>>>,
    func: &mut F,
) where
    F: FnMut(I) -> O,
{
    let mut iterator = dataloader.iter();
    let mut iteration = 0;

    while let Some(item) = iterator.next() {
        let progress = iterator.progress();
        iteration += 1;

        let item = func(item);
        let log = TrainerItem::new(item, progress)
            .iteration(iteration)
            .epoch(epoch)
            .epoch_total(num_epochs);
        logger.log(log);
    }

    logger.clear();
}
