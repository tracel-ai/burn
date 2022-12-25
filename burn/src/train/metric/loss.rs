use super::MetricEntry;
use crate::tensor::backend::Backend;
use crate::tensor::ElementConversion;
use crate::tensor::Tensor;
use crate::train::metric::{Metric, Numeric};

pub struct LossMetric<B: Backend> {
    current: f64,
    count: usize,
    total: f64,
    _b: B,
}

impl<B: Backend> LossMetric<B> {
    pub fn new() -> Self {
        Self {
            count: 0,
            current: 0.0,
            total: 0.0,
            _b: B::default(),
        }
    }
    pub fn reset(&mut self) {
        self.count = 0;
        self.total = 0.0;
        self.current = 0.0;
    }
}

impl<B: Backend> Default for LossMetric<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> Numeric for LossMetric<B> {
    fn value(&self) -> f64 {
        self.current * 100.0
    }
}

impl<B: Backend> Metric for LossMetric<B> {
    type Input = Tensor<B, 1>;

    fn update(&mut self, loss: &Tensor<B, 1>) -> MetricEntry {
        let loss = f64::from_elem(loss.to_data().value[0]);

        self.count += 1;
        self.total += loss;
        self.current = loss;

        let name = String::from("Loss");
        let running = self.total / self.count as f64;
        let raw_current = format!("{}", self.current);
        let formatted = format!("running {:.3} current {:.3}", running, self.current);

        MetricEntry::new(name, formatted, raw_current)
    }

    fn clear(&mut self) {
        self.reset()
    }
}
