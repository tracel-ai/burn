use crate as burn;

use super::LearningRateScheduler;
use crate::{config::Config, LearningRate};

/// Configuration to create a [noam](NoamScheduler) learning rate scheduler.
#[derive(Config)]
pub struct NoamSchedulerConfig {
    /// The initial learning rate.
    learning_rate: LearningRate,
    /// The number of steps before the exponential decay stats.
    #[config(default = 4000)]
    warmup_steps: usize,
    /// The size of the model.
    #[config(default = 512)]
    model_size: usize,
}

/// Noam learning rate scheduler as described in [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
#[derive(Clone, Debug)]
pub struct NoamScheduler {
    warmup_steps: f64,
    embedding_size: f64,
    learning_rate: LearningRate,
    step: f64,
}

impl NoamSchedulerConfig {
    /// Initialize a new [noam](NoamScheduler) learning rate scheduler.
    pub fn init(&self) -> NoamScheduler {
        NoamScheduler {
            warmup_steps: self.warmup_steps as f64,
            embedding_size: self.model_size as f64,
            learning_rate: self.learning_rate,
            step: 0.0,
        }
    }
}

impl LearningRateScheduler for NoamScheduler {
    type Record = usize;

    fn step(&mut self) -> LearningRate {
        self.step += 1.0;

        let arg1 = self.step.powf(-0.5);
        let arg2 = self.step * self.warmup_steps.powf(-1.5);

        self.learning_rate * self.embedding_size.powf(-0.5) * f64::min(arg1, arg2)
    }

    fn to_record(&self) -> Self::Record {
        self.step as usize
    }

    fn load_record(mut self, record: Self::Record) -> Self {
        self.step = record as f64;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_increase_and_decrease() {
        let warmup_steps = 100;
        let mut scheduler = NoamSchedulerConfig::new(10.0)
            .with_warmup_steps(warmup_steps)
            .init();
        let mut lr_current = 0.0;

        for _ in 0..warmup_steps {
            let lr = scheduler.step();
            assert!(
                lr > lr_current,
                "Learning rate should increase before the warmup_steps is reached."
            );
            lr_current = lr;
        }

        for _ in 0..warmup_steps {
            let lr = scheduler.step();
            assert!(
                lr < lr_current,
                "Learning rate should decrease after the warmup_steps is reached."
            );
            lr_current = lr;
        }
    }
}
