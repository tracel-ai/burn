use crate::tensor::back::Backend;
use crate::tensor::ElementConversion;
use crate::tensor::Tensor;

pub trait RunningMetric<I> {
    fn update(&mut self, item: &I) -> String;
    fn clear(&mut self);
}

pub struct LossMetric {
    current: f64,
    count: usize,
    total: f64,
}

impl LossMetric {
    pub fn new() -> Self {
        Self {
            count: 0,
            current: 0.0,
            total: 0.0,
        }
    }
    pub fn update_<B: Backend>(&mut self, loss: &Tensor<B, 1>) -> String {
        let loss = f64::from_elem(loss.to_data().value[0]);

        self.count += 1;
        self.total += loss;
        self.current = loss;

        format!(
            "Loss: current {:.3} batch {:.3}",
            self.total / self.count as f64,
            self.current,
        )
    }

    pub fn clear_(&mut self) {
        self.count = 0;
        self.total = 0.0;
        self.current = 0.0;
    }
}
