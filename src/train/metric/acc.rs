use super::RunningMetricResult;
use crate::tensor::back::Backend;
use crate::tensor::ElementConversion;
use crate::tensor::Tensor;
use crate::train::metric::RunningMetric;

pub struct AccMetric {
    current: f64,
    count: usize,
    total: f64,
}

impl AccMetric {
    pub fn new() -> Self {
        Self {
            count: 0,
            current: 0.0,
            total: 0.0,
        }
    }
}
impl<B: Backend> RunningMetric<(Tensor<B, 2>, Tensor<B, 2>)> for AccMetric {
    fn update(&mut self, batch: &(Tensor<B, 2>, Tensor<B, 2>)) -> RunningMetricResult {
        let (outputs, targets) = batch;
        // TODO: Needs Argmax

        self.count += 1;
        self.total += 1.0;
        self.current = 1.0;

        let name = String::from("Acc");
        let running = self.total / self.count as f64;
        let raw_running = format!("{}", running);
        let raw_current = format!("{}", self.current);
        let formatted = format!("running {:.3} current {:.3}", running, self.current);

        RunningMetricResult {
            name,
            formatted,
            raw_running,
            raw_current,
        }
    }

    fn clear(&mut self) {
        self.count = 0;
        self.total = 0.0;
        self.current = 0.0;
    }
}
