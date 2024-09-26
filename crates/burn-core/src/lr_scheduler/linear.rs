use super::LrScheduler;
use crate as burn;
use crate::{config::Config, LearningRate};
use burn_tensor::backend::Backend;

/// The configuration for creating a linear learning rate scheduler.
///
/// This scheduler starts at a learning rate `initial_lr`, then changes the learning rate by a constant amount on each
/// iteration until reaching a final learning rate `final_lr`. The `num_iters` parameter controls how many iterations
/// are needed to go from `initial_lr` to `final_lr`.
#[derive(Config)]
pub struct LinearLrSchedulerConfig {
    // The initial learning rate.
    initial_lr: LearningRate,
    // The final learning rate.
    final_lr: LearningRate,
    // The number of iterations before reaching the final learning rate.
    num_iters: usize,
}

impl LinearLrSchedulerConfig {
    /// Initializes a [linear learning rate scheduler](LinearLrScheduler).
    ///
    /// # Panics
    /// This function panics if `initial_lr` and `final_lr` are not between 0 and 1.
    pub fn init(&self) -> LinearLrScheduler {
        assert!(
            self.initial_lr > 0. && self.initial_lr <= 1.,
            "Initial learning rate must be greater than 0 and at most 1"
        );
        assert!(
            self.final_lr >= 0. && self.final_lr <= 1.,
            "Final learning rate must be at least 0 and at most 1"
        );

        LinearLrScheduler {
            previous_lr: self.initial_lr,
            step_size: (self.final_lr - self.initial_lr) / self.num_iters as f64,
            remaining_iters: self.num_iters,
        }
    }
}

/// A linear learning rate scheduler.
///
/// See [LinearLrSchedulerConfig] for more information.
#[derive(Clone, Copy, Debug)]
pub struct LinearLrScheduler {
    // The previous iteration's learning rate.
    previous_lr: LearningRate,
    // The amount that the learning rate changes by on each iteration.
    step_size: f64,
    // The number of iterations left before reaching the final learning rate.
    remaining_iters: usize,
}

impl LrScheduler for LinearLrScheduler {
    type Record<B: Backend> = (LearningRate, f64, usize);

    fn step(&mut self) -> LearningRate {
        if self.remaining_iters > 0 {
            self.remaining_iters -= 1;
            self.previous_lr += self.step_size;
        }
        self.previous_lr
    }

    fn to_record<B: Backend>(&self) -> Self::Record<B> {
        (self.previous_lr, self.step_size, self.remaining_iters)
    }

    fn load_record<B: Backend>(mut self, record: Self::Record<B>) -> Self {
        (self.previous_lr, self.step_size, self.remaining_iters) = record;
        self
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    #[should_panic = "Initial learning rate must be greater than 0 and at most 1"]
    fn config_initial_lr_too_low() {
        LinearLrSchedulerConfig::new(0., 0.5, 100).init();
    }

    #[test]
    #[should_panic = "Initial learning rate must be greater than 0 and at most 1"]
    fn config_initial_lr_too_high() {
        LinearLrSchedulerConfig::new(1.5, 0.5, 100).init();
    }

    #[test]
    #[should_panic = "Final learning rate must be at least 0 and at most 1"]
    fn config_final_lr_too_low() {
        LinearLrSchedulerConfig::new(0.5, -0.5, 100).init();
    }

    #[test]
    #[should_panic = "Final learning rate must be at least 0 and at most 1"]
    fn config_final_lr_too_high() {
        LinearLrSchedulerConfig::new(0.5, 1.5, 100).init();
    }

    #[test]
    fn test_lr_change() {
        const INITIAL_LR: LearningRate = 0.75;
        const NUM_ITERS: usize = 10;

        let mut scheduler = LinearLrSchedulerConfig::new(INITIAL_LR, 0.25, NUM_ITERS).init();

        let mut previous_lr = INITIAL_LR;

        for _ in 0..NUM_ITERS {
            let lr = scheduler.step();
            assert!(
                lr < previous_lr,
                "Learning rate should decrease with each iteration before reaching the final learning rate"
            );
            previous_lr = lr;
        }

        for _ in 0..NUM_ITERS {
            let lr = scheduler.step();
            assert_eq!(
                previous_lr, lr,
                "Learning rate should remain constant after reaching the final learning rate"
            )
        }
    }
}
