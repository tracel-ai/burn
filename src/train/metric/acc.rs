use super::RunningMetricResult;
use crate::tensor::back::Backend;
use crate::tensor::Tensor;
use crate::train::metric::RunningMetric;

pub struct AccuracyMetric {
    current: f64,
    count: usize,
    total: usize,
}

impl AccuracyMetric {
    pub fn new() -> Self {
        Self {
            count: 0,
            current: 0.0,
            total: 0,
        }
    }
}

impl<B: Backend> RunningMetric<(Tensor<B, 2>, Tensor<B, 2>)> for AccuracyMetric {
    fn update(&mut self, batch: &(Tensor<B, 2>, Tensor<B, 2>)) -> RunningMetricResult {
        let (outputs, targets) = batch;
        let logits_outputs = outputs.argmax(1).to_data();
        let logits_targets = targets.argmax(1).to_data();

        let mut total_current = 0;

        for (output, target) in logits_outputs.value.iter().zip(logits_targets.value.iter()) {
            if output == target {
                total_current += 1;
            }
        }

        let count_current = targets.shape().dims[0];

        self.count += count_current;
        self.total += total_current;
        self.current = total_current as f64 / count_current as f64;

        let name = String::from("Accurracy");
        let running = self.total as f64 / self.count as f64;
        let raw_running = format!("{}", running);
        let raw_current = format!("{}", self.current);
        let formatted = format!(
            "running {:.2} % current {:.2} %",
            100.0 * running,
            100.0 * self.current
        );

        RunningMetricResult {
            name,
            formatted,
            raw_running,
            raw_current,
        }
    }

    fn clear(&mut self) {
        self.count = 0;
        self.total = 0;
        self.current = 0.0;
    }
}
