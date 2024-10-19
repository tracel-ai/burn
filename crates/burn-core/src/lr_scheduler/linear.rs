use super::LrScheduler;
use crate as burn;
use crate::{config::Config, LearningRate};
use burn_tensor::backend::Backend;

/// The configuration for creating a [linear learning rate scheduler](LinearLrScheduler).
///
/// This scheduler returns the learning rate `initial_lr` at the first step, then changes it by a
/// constant amount on each iteration until reaching a final learning rate `final_lr`. The
/// `num_iters` parameter controls how many iterations are needed to go from `initial_lr` to
/// `final_lr`.
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
        assert!(
            self.num_iters > 0,
            "Number of iterations must be at least 1"
        );

        LinearLrScheduler {
            final_lr: self.final_lr,
            step_size: (self.final_lr - self.initial_lr) / self.num_iters as f64,
            remaining_iters: self.num_iters + 1,
        }
    }
}

/// A linear learning rate scheduler.
///
/// See [LinearLrSchedulerConfig] for more information.
#[derive(Clone, Copy, Debug)]
pub struct LinearLrScheduler {
    // The final learning rate after the linear changing process stops.
    final_lr: LearningRate,
    // The amount that the learning rate changes by on each iteration.
    step_size: f64,
    // The number of iterations left before reaching the final learning rate.
    remaining_iters: usize,
}

impl LrScheduler for LinearLrScheduler {
    type Record<B: Backend> = usize;

    fn step(&mut self) -> LearningRate {
        self.remaining_iters -= (self.remaining_iters != 0) as usize;
        self.final_lr - self.step_size * self.remaining_iters as f64
    }

    fn to_record<B: Backend>(&self) -> Self::Record<B> {
        self.remaining_iters
    }

    fn load_record<B: Backend>(mut self, record: Self::Record<B>) -> Self {
        self.remaining_iters = record;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_utils;
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
    #[should_panic = "Number of iterations must be at least 1"]
    fn config_num_iters_too_low() {
        LinearLrSchedulerConfig::new(0.9, 0.1, 0).init();
    }

    #[test]
    fn test_lr_decreasing() {
        const INITIAL_LR: LearningRate = 0.9;
        const FINAL_LR: LearningRate = 0.5;
        const NUM_ITERS: usize = 4;
        let scheduler = LinearLrSchedulerConfig::new(INITIAL_LR, FINAL_LR, NUM_ITERS).init();
        let expected_lrs = [0.9, 0.8, 0.7, 0.6, 0.5, 0.5];
        test_utils::check_lr_sequence(scheduler, expected_lrs);
    }

    #[test]
    fn test_lr_increasing() {
        const INITIAL_LR: LearningRate = 0.01;
        const FINAL_LR: LearningRate = 0.04;
        const NUM_ITERS: usize = 3;
        let scheduler = LinearLrSchedulerConfig::new(INITIAL_LR, FINAL_LR, NUM_ITERS).init();
        let expected_lrs = [0.01, 0.02, 0.03, 0.04, 0.04];
        test_utils::check_lr_sequence(scheduler, expected_lrs);
    }

    #[test]
    fn test_lr_unchanging() {
        const INITIAL_LR: LearningRate = 0.3;
        const FINAL_LR: LearningRate = 0.3;
        const NUM_ITERS: usize = 2;
        let scheduler = LinearLrSchedulerConfig::new(INITIAL_LR, FINAL_LR, NUM_ITERS).init();
        let expected_lrs = [0.3, 0.3, 0.3, 0.3];
        test_utils::check_lr_sequence(scheduler, expected_lrs);
    }

    #[test]
    fn test_save_and_load() {
        const INITIAL_LR: LearningRate = 1.0;
        const FINAL_LR: LearningRate = 0.01;
        const NUM_ITERS: usize = 6;
        let scheduler = LinearLrSchedulerConfig::new(INITIAL_LR, FINAL_LR, NUM_ITERS).init();
        test_utils::check_save_load(scheduler, NUM_ITERS / 3 * 2);
    }
}
