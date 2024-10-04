use super::LrScheduler;
use crate as burn;
use crate::{config::Config, LearningRate};
use burn_tensor::backend::Backend;

/// The configuration for creating a [Cosine Annealing learning rate scheduler with warm
/// restarts](CosineAnnealingLrScheduler).
///
/// This scheduler returns the learning rate `initial_lr` at the first step, then changes it by
/// following a cosine function. After `num_iters` iterations, the learning rate is reset to
/// `initial_lr`.
#[derive(Config)]
pub struct CosineAnnealingLrSchedulerConfig {
    // The initial learning rate.
    initial_lr: LearningRate,
    // The final learning rate.
    #[config(default = 0.0)]
    min_lr: LearningRate,
    // The number of iterations between two restarts. The two restart iterations themselves are not
    // included.
    num_iters: usize,
}

impl CosineAnnealingLrSchedulerConfig {
    /// Initializes a [Cosine learning rate scheduler](CosineAnnealingLrScheduler).
    ///
    /// # Panics
    /// This function panics if `initial_lr` and `final_lr` are not between 0 and 1.
    pub fn init(&self) -> CosineAnnealingLrScheduler {
        assert!(
            self.initial_lr > 0. && self.initial_lr <= 1.,
            "Initial learning rate must be greater than 0 and at most 1"
        );
        assert!(
            self.min_lr >= 0.0 && self.min_lr <= self.initial_lr,
            "Minimum learning rate must be at least 0 and at most equal to the initial learning rate"
        );
        assert!(
            self.num_iters > 0,
            "Number of iterations must be at least 1"
        );

        CosineAnnealingLrScheduler {
            min_lr: self.min_lr,
            max_lr: self.initial_lr,
            num_iters: self.num_iters,
            current_iter: usize::MAX,
        }
    }
}

/// A Cosine Annealing learning rate scheduler.
///
/// This scheduler is described in [SGDR: Stochastic Gradient Descent with Warm
/// Restarts](https://arxiv.org/abs/1608.03983). See [CosineAnnealingLrSchedulerConfig] for more
/// information.
#[derive(Clone, Copy, Debug)]
pub struct CosineAnnealingLrScheduler {
    min_lr: LearningRate,
    max_lr: LearningRate,
    num_iters: usize,
    current_iter: usize,
}

impl LrScheduler for CosineAnnealingLrScheduler {
    type Record<B: Backend> = usize;

    fn step(&mut self) -> LearningRate {
        // Make current_iter overflow from usize::MAX to 0 to get the initial learning rate on the
        // first call. We could've used i64 with an initial value -1, but keeping it in usize saves
        // us from some type casting here.
        self.current_iter = self.current_iter.wrapping_add(1) % (self.num_iters + 1);
        self.min_lr
            + 0.5
                * (self.max_lr - self.min_lr)
                * (1.0
                    + (self.current_iter as f64 / self.num_iters as f64 * std::f64::consts::PI)
                        .cos())
    }

    fn to_record<B: Backend>(&self) -> Self::Record<B> {
        self.current_iter
    }

    fn load_record<B: Backend>(mut self, record: Self::Record<B>) -> Self {
        self.current_iter = record;
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
        CosineAnnealingLrSchedulerConfig::new(0., 10).init();
    }

    #[test]
    #[should_panic = "Initial learning rate must be greater than 0 and at most 1"]
    fn config_initial_lr_too_high() {
        CosineAnnealingLrSchedulerConfig::new(1.5, 10).init();
    }

    #[test]
    #[should_panic = "Minimum learning rate must be at least 0 and at most equal to the initial learning rate"]
    fn config_min_lr_too_low() {
        CosineAnnealingLrSchedulerConfig::new(0.5, 10)
            .with_min_lr(-0.1)
            .init();
    }

    #[test]
    #[should_panic = "Minimum learning rate must be at least 0 and at most equal to the initial learning rate"]
    fn config_min_lr_too_high() {
        CosineAnnealingLrSchedulerConfig::new(0.5, 10)
            .with_min_lr(0.6)
            .init();
    }

    #[test]
    #[should_panic = "Number of iterations must be at least 1"]
    fn config_num_iters_too_low() {
        CosineAnnealingLrSchedulerConfig::new(0.5, 0).init();
    }

    #[test]
    fn test_lr_change() {
        const INITIAL_LR: LearningRate = 0.5;
        const MIN_LR: LearningRate = 0.1;
        const NUM_ITERS: usize = 2;

        let scheduler = CosineAnnealingLrSchedulerConfig::new(INITIAL_LR, NUM_ITERS)
            .with_min_lr(MIN_LR)
            .init();
        let expected_lrs = [
            INITIAL_LR,                  // cos(0)
            (INITIAL_LR + MIN_LR) * 0.5, // cos(PI/2)
            MIN_LR,                      // cos(PI)
            INITIAL_LR,                  // restart
        ];
        test_utils::check_lr_sequence(scheduler, expected_lrs);
    }

    #[test]
    fn test_save_and_load() {
        const INITIAL_LR: LearningRate = 1.0;
        const NUM_ITERS: usize = 9;
        let scheduler = CosineAnnealingLrSchedulerConfig::new(INITIAL_LR, NUM_ITERS).init();
        test_utils::check_save_load(scheduler, NUM_ITERS / 3 * 2);
    }
}
