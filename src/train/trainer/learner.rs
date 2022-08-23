use crate::module::Module;
use crate::optim::Optimizer;
use crate::tensor::back::{ad, Backend};
use burn_tensor::Tensor;

pub trait Loss<B: Backend, T>: Module<Backend = B> {
    fn loss(&self, item: T) -> Tensor<B, 1>;
}

pub trait Learner<B: Backend, T, V, O, TO, VO> {
    fn train(&mut self, item: T, epoch: usize, optim: &mut O) -> TO;
    fn valid(&self, item: V, epoch: usize) -> VO;
    fn test(&self, item: V) -> VO;
}

#[derive(new)]
pub struct SimpleLearner<L> {
    model: L,
}

#[derive(new)]
pub struct SimpleOutput<B: Backend> {
    pub loss: Tensor<B, 1>,
}

impl<B, T, L, O> Learner<B, T, T, O, SimpleOutput<B>, SimpleOutput<B>> for SimpleLearner<L>
where
    B: ad::Backend,
    L: Loss<B, T>,
    O: Optimizer<B>,
{
    fn train(&mut self, item: T, epoch: usize, optim: &mut O) -> SimpleOutput<B> {
        let loss = self.model.loss(item);
        let grads = loss.backward();

        self.model.update_params(&grads, optim);

        SimpleOutput::new(loss)
    }

    fn valid(&self, item: T, epoch: usize) -> SimpleOutput<B> {
        let loss = self.model.loss(item);
        SimpleOutput::new(loss)
    }

    fn test(&self, item: T) -> SimpleOutput<B> {
        let loss = self.model.loss(item);
        SimpleOutput::new(loss)
    }
}
