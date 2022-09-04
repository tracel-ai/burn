use super::{Learner, Loss};
use crate::module::ADModule;
use crate::optim::Optimizer;
use crate::tensor::backend::{ADBackend, Backend};
use crate::train::metric;
use burn_tensor::Tensor;

#[derive(new)]
pub struct BasicLearner<L, O> {
    model: L,
    optim: O,
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

impl<B, B2, T, L, L2, O> Learner<T, T, BasicOutput<B>, BasicOutput<B2>> for BasicLearner<L, O>
where
    B: ADBackend<InnerBackend = B2>,
    B2: Backend,
    O: Optimizer<Backend = B>,
    L: Loss<Backend = B, Item = T> + ADModule<Backend = B, InnerModule = L2>,
    L2: Loss<Backend = B::InnerBackend, Item = T>,
    O: Optimizer<Backend = B>,
{
    type Backend = B;

    fn train(&mut self, item: T) -> BasicOutput<B> {
        let loss = self.model.loss(item);
        let grads = loss.backward();

        self.model.update_params(&grads, &mut self.optim);

        BasicOutput::new(loss)
    }

    fn valid(&self, item: T) -> BasicOutput<B2> {
        let loss = self.model.inner().loss(item);
        BasicOutput::new(loss)
    }
}
