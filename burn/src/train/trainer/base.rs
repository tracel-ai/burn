use super::{Learner, TrainerItem};
use crate::data::dataloader::DataLoader;
use crate::tensor::backend::ADBackend;
use crate::train::logger::TrainValidLogger;
use std::sync::Arc;

pub struct SupervisedTrainer<B, T, V, L, TO, VO>
where
    B: ADBackend,
    L: Learner<T, V, TO, VO, Backend = B>,
{
    dataloader_train: Arc<dyn DataLoader<T>>,
    dataloader_valid: Arc<dyn DataLoader<V>>,
    logger: Box<dyn TrainValidLogger<TrainerItem<TO>, TrainerItem<VO>>>,
    learner: L,
    _b: B,
}

impl<B, T, V, L, TO, VO> SupervisedTrainer<B, T, V, L, TO, VO>
where
    B: ADBackend,
    L: Learner<T, V, TO, VO, Backend = B>,
{
    pub fn new(
        dataloader_train: Arc<dyn DataLoader<T>>,
        dataloader_valid: Arc<dyn DataLoader<V>>,
        logger: Box<dyn TrainValidLogger<TrainerItem<TO>, TrainerItem<VO>>>,
        learner: L,
    ) -> Self {
        Self {
            dataloader_train,
            dataloader_valid,
            learner,
            logger,
            _b: B::default(),
        }
    }

    pub fn run(mut self, num_epochs: usize) -> L {
        for epoch in 0..num_epochs {
            run_step(
                epoch,
                num_epochs,
                &self.dataloader_train,
                &mut |item| self.learner.train(item),
                &mut |log| self.logger.log_train(log),
            );
            self.logger.clear_train();

            run_step(
                epoch,
                num_epochs,
                &self.dataloader_valid,
                &mut |item| self.learner.valid(item),
                &mut |log| self.logger.log_valid(log),
            );
            self.logger.clear_valid();
        }

        self.learner
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
