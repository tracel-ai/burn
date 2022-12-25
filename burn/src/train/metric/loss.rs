use super::RunningMetricResult;
use crate::tensor::backend::Backend;
use crate::tensor::ElementConversion;
use crate::tensor::Tensor;
use crate::train::metric::{Metric, MetricState, Numeric};

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
}

impl Default for LossMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl Numeric for LossMetric {
    fn value(&self) -> f64 {
        self.current * 100.0
    }
}

impl<B: Backend> Metric<Tensor<B, 1>> for LossMetric {
    fn update(&mut self, loss: &Tensor<B, 1>) -> Box<dyn MetricState> {
        let loss = f64::from_elem(loss.to_data().value[0]);

        self.count += 1;
        self.total += loss;
        self.current = loss;

        let name = String::from("Loss");
        let running = self.total / self.count as f64;
        let raw_running = format!("{running}");
        let raw_current = format!("{}", self.current);
        let formatted = format!("running {:.3} current {:.3}", running, self.current);

        Box::new(RunningMetricResult {
            name,
            formatted,
            raw_running,
            raw_current,
        })
    }

    fn clear(&mut self) {
        self.count = 0;
        self.total = 0.0;
        self.current = 0.0;
    }
}
