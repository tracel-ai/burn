use super::{Learner, Loss};
use crate::optim::Optimizer;
use crate::tensor::back::{ad, Backend};
use crate::train::metric;
use burn_tensor::Tensor;

#[derive(new)]
pub struct BasicLearner<L> {
    model: L,
}

#[derive(new)]
pub struct BasicOutput<B: Backend> {
    pub loss: Tensor<B, 1>,
}

impl<B: Backend> metric::Metric<BasicOutput<B>> for metric::LossMetric {
    fn update(&mut self, item: &BasicOutput<B>) -> metric::MetricStateDyn {
        self.update(&item.loss)
    }
    fn clear(&mut self) {
        <metric::LossMetric as metric::Metric<Tensor<B, 1>>>::clear(self);
    }
}

impl<B, T, L, O> Learner<B, T, T, O, BasicOutput<B>, BasicOutput<B>> for BasicLearner<L>
where
    B: ad::Backend,
    L: Loss<B, T>,
    O: Optimizer<B>,
{
    fn train(&mut self, item: T, optim: &mut O) -> BasicOutput<B> {
        let loss = self.model.loss(item);
        let grads = loss.backward();

        self.model.update_params(&grads, optim);

        BasicOutput::new(loss)
    }

    fn valid(&self, item: T) -> BasicOutput<B> {
        let loss = self.model.loss(item);
        BasicOutput::new(loss)
    }

    fn test(&self, item: T) -> BasicOutput<B> {
        let loss = self.model.loss(item);
        BasicOutput::new(loss)
    }
}
