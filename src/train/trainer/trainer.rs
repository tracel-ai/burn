use super::Learner;
use crate::data::dataloader::DataLoader;
use crate::module::Module;
use crate::optim::Optimizer;
use crate::tensor::back::ad;
use std::sync::Arc;

pub struct SupervisedTrainer<B, T, V, L, O>
where
    B: ad::Backend,
    L: Learner<B, T, V, O>,
    O: Optimizer<B>,
{
    dataloader_train: Arc<dyn DataLoader<T>>,
    dataloader_valid: Arc<dyn DataLoader<V>>,
    dataloader_test: Arc<dyn DataLoader<V>>,
    learner: L,
    optimizer: O,
    _b: B,
}

impl<B, T, V, L, O> SupervisedTrainer<B, T, V, L, O>
where
    B: ad::Backend,
    L: Learner<B, T, V, O>,
    O: Optimizer<B>,
{
    pub fn new(
        dataloader_train: Arc<dyn DataLoader<T>>,
        dataloader_valid: Arc<dyn DataLoader<V>>,
        dataloader_test: Arc<dyn DataLoader<V>>,
        learner: L,
        optimizer: O,
    ) -> Self {
        Self {
            dataloader_train,
            dataloader_valid,
            dataloader_test,
            learner,
            optimizer,
            _b: B::default(),
        }
    }

    pub fn run(mut self, num_epochs: usize) -> L {
        let dataloader_train = self.dataloader_train.clone();
        let dataloader_valid = self.dataloader_valid.clone();

        for epoch in 0..num_epochs {
            for item in dataloader_train.iter() {
                self.learner.train(item, epoch, &mut self.optimizer);
            }

            for item in dataloader_valid.iter() {
                self.learner.valid(item, epoch);
            }
        }

        let dataloader_test = self.dataloader_test.clone();
        for item in dataloader_test.iter() {
            self.learner.test(item);
        }

        self.learner
    }
}
