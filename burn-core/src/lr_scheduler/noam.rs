use crate as burn;

use super::{LearningRate, LearningRateScheduler};
use crate::record::Record;

#[derive(new, Record, Clone, Debug)]
pub struct NoamScheduler {
    warmup_steps: usize,
    embedding_size: usize,
    learning_rate: LearningRate,
    step: usize,
}

impl LearningRateScheduler for NoamScheduler {
    type Record = Self;

    fn step(&mut self) -> LearningRate {
        self.step += 1;

        let arg1 = rsqrt(self.step as f64);
        let arg2 = self.step as f64 * (self.warmup_steps as f64).powf(-1.5);

        self.learning_rate * rsqrt(self.embedding_size as f64) * f64::min(arg1, arg2)
    }
    fn to_record(&self) -> Self::Record {
        self.clone()
    }
}

fn rsqrt(value: f64) -> f64 {
    return 1.0 / (value.sqrt());
}
