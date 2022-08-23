use super::Learner;
use crate::module::{Forward, Module};
use crate::optim::Optimizer;
use crate::tensor::back::{ad, Backend};
use crate::train::metric;
use burn_tensor::Tensor;

#[derive(new)]
pub struct ClassificationLearner<M> {
    model: M,
}

#[derive(new)]
pub struct ClassificationOutput<B: Backend> {
    pub loss: Tensor<B, 1>,
    pub output: Tensor<B, 2>,
    pub targets: Tensor<B, 2>,
}

impl<B: Backend> metric::RunningMetric<ClassificationOutput<B>> for metric::LossMetric {
    fn update(&mut self, item: &ClassificationOutput<B>) -> metric::RunningMetricResult {
        self.update(&item.loss)
    }
    fn clear(&mut self) {
        <metric::LossMetric as metric::RunningMetric<Tensor<B, 1>>>::clear(self);
    }
}

impl<B, I, M, O> Learner<B, I, I, O, ClassificationOutput<B>, ClassificationOutput<B>>
    for ClassificationLearner<M>
where
    B: ad::Backend,
    M: Forward<I, ClassificationOutput<B>> + Module<Backend = B>,
    O: Optimizer<B>,
{
    fn train(&mut self, item: I, optim: &mut O) -> ClassificationOutput<B> {
        let output = self.model.forward(item);
        let grads = output.loss.backward();

        self.model.update_params(&grads, optim);

        output
    }

    fn valid(&self, item: I) -> ClassificationOutput<B> {
        self.model.forward(item)
    }

    fn test(&self, item: I) -> ClassificationOutput<B> {
        self.model.forward(item)
    }
}
