use super::Learner;
use crate::data::dataloader::DataLoader;
use crate::optim::Optimizer;
use crate::tensor::back::ad;
use crate::train::logger::{LogItem, Logger};
use std::sync::Arc;

pub struct SupervisedTrainer<B, T, V, L, O, TO, VO>
where
    B: ad::Backend,
    L: Learner<B, T, V, O, TO, VO>,
    O: Optimizer<B>,
{
    dataloader_train: Arc<dyn DataLoader<T>>,
    dataloader_valid: Arc<dyn DataLoader<V>>,
    dataloader_test: Arc<dyn DataLoader<V>>,
    logger_train: Box<dyn Logger<TO>>,
    logger_valid: Box<dyn Logger<VO>>,
    logger_test: Box<dyn Logger<VO>>,
    learner: L,
    optimizer: O,
    _b: B,
}

impl<B, T, V, L, O, TO, VO> SupervisedTrainer<B, T, V, L, O, TO, VO>
where
    B: ad::Backend,
    L: Learner<B, T, V, O, TO, VO>,
    O: Optimizer<B>,
{
    pub fn new(
        dataloader_train: Arc<dyn DataLoader<T>>,
        dataloader_valid: Arc<dyn DataLoader<V>>,
        dataloader_test: Arc<dyn DataLoader<V>>,
        logger_train: Box<dyn Logger<TO>>,
        logger_valid: Box<dyn Logger<VO>>,
        logger_test: Box<dyn Logger<VO>>,
        learner: L,
        optimizer: O,
    ) -> Self {
        Self {
            dataloader_train,
            dataloader_valid,
            dataloader_test,
            learner,
            optimizer,
            logger_train,
            logger_valid,
            logger_test,
            _b: B::default(),
        }
    }

    pub fn run(mut self, num_epochs: usize) -> L {
        for epoch in 0..num_epochs {
            let mut iterator = self.dataloader_train.iter();
            let mut iteration = 0;

            while let Some(item) = iterator.next() {
                let progress = iterator.progress();
                iteration += 1;

                let item = self.learner.train(item, &mut self.optimizer);
                let log = LogItem::new(item, progress)
                    .iteration(iteration)
                    .epoch(epoch)
                    .epoch_total(num_epochs);
                self.logger_train.log(log);
            }

            self.logger_train.clear();

            let mut iterator = self.dataloader_valid.iter();
            let mut iteration = 0;

            while let Some(item) = iterator.next() {
                let progress = iterator.progress();
                iteration += 1;

                let item = self.learner.valid(item);
                let log = LogItem::new(item, progress)
                    .iteration(iteration)
                    .epoch(epoch)
                    .epoch_total(num_epochs);
                self.logger_valid.log(log);
            }

            self.logger_valid.clear();
        }

        let mut iterator = self.dataloader_test.iter();
        let mut iteration = 0;

        while let Some(item) = iterator.next() {
            let progress = iterator.progress();
            iteration += 1;

            let item = self.learner.test(item);
            let log = LogItem::new(item, progress).iteration(iteration);
            self.logger_test.log(log);
        }

        self.logger_test.clear();

        self.learner
    }
}
