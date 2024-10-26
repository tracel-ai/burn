use burn_tensor::backend::Backend;

use crate as burn;

use super::LrScheduler;
use crate::{config::Config, LearningRate};

/// Configuration to create a [noam](NoamLrScheduler) learning rate scheduler.
#[derive(Config)]
pub struct NoamLrSchedulerConfig {
    /// The initial learning rate.
    init_lr: LearningRate,
    /// The number of steps before the exponential decay stats.
    #[config(default = 4000)]
    warmup_steps: usize,
    /// The size of the model.
    #[config(default = 512)]
    model_size: usize,
}

/// Noam learning rate scheduler as described in [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
#[derive(Clone, Debug)]
pub struct NoamLrScheduler {
    warmup_steps: f64,
    embedding_size: f64,
    init_lr: LearningRate,
    step: f64,
}

impl NoamLrSchedulerConfig {
    /// Initialize a new [noam](NoamLrScheduler) learning rate scheduler.
    pub fn init(&self) -> NoamLrScheduler {
        NoamLrScheduler {
            warmup_steps: self.warmup_steps as f64,
            embedding_size: self.model_size as f64,
            init_lr: self.init_lr,
            step: 0.0,
        }
    }
}

impl LrScheduler for NoamLrScheduler {
    type Record<B: Backend> = usize;

    fn step(&mut self) -> LearningRate {
        self.step += 1.0;

        let arg1 = self.step.powf(-0.5);
        let arg2 = self.step * self.warmup_steps.powf(-1.5);

        self.init_lr * self.embedding_size.powf(-0.5) * f64::min(arg1, arg2)
    }

    fn to_record<B: Backend>(&self) -> Self::Record<B> {
        self.step as usize
    }

    fn load_record<B: Backend>(mut self, record: Self::Record<B>) -> Self {
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
        let mut scheduler = NoamLrSchedulerConfig::new(10.0)
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
