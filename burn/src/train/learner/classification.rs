use crate::tensor::backend::Backend;
use crate::train::metric;
use burn_tensor::Tensor;

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
