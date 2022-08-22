use crate::module::Module;
use crate::optim::Optimizer;
use crate::tensor::back::{ad, Backend};
use burn_tensor::Tensor;

pub trait Loss<B: Backend, T>: Module<Backend = B> {
    fn loss(&self, item: T) -> Tensor<B, 1>;
}

pub trait Learner<B: Backend, T, V, O> {
    fn train(&mut self, item: T, epoch: usize, optim: &mut O);
    fn valid(&self, item: V, epoch: usize);
    fn test(&self, item: V);
}

#[derive(new)]
pub struct SimpleLearner<L> {
    model: L,
}

impl<B, T, L, O> Learner<B, T, T, O> for SimpleLearner<L>
where
    B: ad::Backend,
    L: Loss<B, T>,
    O: Optimizer<B>,
{
    fn train(&mut self, item: T, epoch: usize, optim: &mut O) {
        let loss = self.model.loss(item);
        let grads = loss.backward();

        self.model.update_params(&grads, optim);
        println!("Train | Epoch {} - Loss {}", epoch, loss.to_data());
    }

    fn valid(&self, item: T, epoch: usize) {
        let loss = self.model.loss(item);
        println!("Valid | Epoch {} - Loss {}", epoch, loss.to_data());
    }

    fn test(&self, item: T) {
        let loss = self.model.loss(item);
        println!("Test | Loss {}", loss.to_data());
    }
}
