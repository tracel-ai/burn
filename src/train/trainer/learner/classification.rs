use super::Learner;
use crate::module::{ADModule, Forward, Module};
use crate::optim::Optimizer;
use crate::tensor::backend::{ADBackend, Backend};
use crate::train::metric;
use burn_tensor::Tensor;

#[derive(new)]
pub struct ClassificationLearner<M, O> {
    pub model: M,
    optim: O,
}

#[derive(new)]
pub struct ClassificationOutput<B: Backend> {
    pub loss: Tensor<B, 1>,
    pub output: Tensor<B, 2>,
    pub targets: Tensor<B, 2>,
}

impl<B: Backend> metric::Metric<ClassificationOutput<B>> for metric::LossMetric {
    fn update(&mut self, item: &ClassificationOutput<B>) -> metric::MetricStateDyn {
        self.update(&item.loss)
    }
    fn clear(&mut self) {
        <metric::LossMetric as metric::Metric<Tensor<B, 1>>>::clear(self);
    }
}

impl<B: Backend> metric::Metric<ClassificationOutput<B>> for metric::AccuracyMetric {
    fn update(&mut self, item: &ClassificationOutput<B>) -> metric::MetricStateDyn {
        self.update(&(item.output.clone(), item.targets.clone()))
    }

    fn clear(&mut self) {
        <metric::AccuracyMetric as metric::Metric<(Tensor<B, 2>, Tensor<B, 2>)>>::clear(self);
    }
}

impl<B, I, IV, M, M2, O>
    Learner<I, IV, ClassificationOutput<B>, ClassificationOutput<B::InnerBackend>>
    for ClassificationLearner<M, O>
where
    B: ADBackend,
    M: Forward<I, ClassificationOutput<B>> + ADModule<Backend = B, InnerModule = M2>,
    M2: Forward<IV, ClassificationOutput<B::InnerBackend>> + Module<Backend = B::InnerBackend>,
    O: Optimizer<Backend = B>,
{
    type Backend = B;

    fn train(&mut self, item: I) -> ClassificationOutput<B> {
        let output = self.model.forward(item);
        let grads = output.loss.backward();

        self.model.update_params(&grads, &mut self.optim);

        output
    }

    fn valid(&self, item: IV) -> ClassificationOutput<B::InnerBackend> {
        self.model.inner().forward(item)
    }
}
