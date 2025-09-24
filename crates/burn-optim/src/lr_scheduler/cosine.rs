use burn_core as burn;

use super::{LrScheduler, String};
use crate::LearningRate;
use burn::config::Config;
use burn::tensor::backend::Backend;

/// The configuration for creating a [Cosine Annealing learning rate scheduler with warm
/// restarts](CosineAnnealingLrScheduler).
///
/// This scheduler returns the learning rate `initial_lr` at the first step, then changes it by
/// following a cosine function. After `num_iters` iterations, the learning rate is reset to
/// `initial_lr`.
#[derive(Config, Debug)]
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
    /// # Errors
    ///
    /// An error will be returned if any of the following conditions is true:
    ///
    /// * `initial_lr` is out of range (0.0, 1.0]
    /// * `min_lr` is out of range [0.0, `initial_lr`]
    /// * `num_iters` is 0
    pub fn init(&self) -> Result<CosineAnnealingLrScheduler, String> {
        if self.initial_lr <= 0. || self.initial_lr > 1. {
            return Err("Initial learning rate must be greater than 0 and at most 1".into());
        }
        if self.min_lr < 0.0 || self.min_lr > self.initial_lr {
            return Err(
                "Minimum learning rate must be at least 0 and at most equal to the initial \
                 learning rate"
                    .into(),
            );
        }
        if self.num_iters == 0 {
            return Err("Number of iterations must be at least 1".into());
        }

        Ok(CosineAnnealingLrScheduler {
            min_lr: self.min_lr,
            max_lr: self.initial_lr,
            num_iters: self.num_iters,
            current_iter: usize::MAX,
        })
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
    fn config_initial_lr_too_low() {
        let r = CosineAnnealingLrSchedulerConfig::new(0., 10).init();
        assert!(r.is_err(), "Should return an error");
        assert_eq!(
            r.unwrap_err(),
            "Initial learning rate must be greater than 0 and at most 1",
            "Error messages should match",
        );
    }

    #[test]
    fn config_initial_lr_too_high() {
        let r = CosineAnnealingLrSchedulerConfig::new(1.5, 10).init();
        assert!(r.is_err(), "Should return an error");
        assert_eq!(
            r.unwrap_err(),
            "Initial learning rate must be greater than 0 and at most 1",
            "Error messages should match",
        );
    }

    #[test]
    fn config_min_lr_too_low() {
        let r = CosineAnnealingLrSchedulerConfig::new(0.5, 10)
            .with_min_lr(-0.1)
            .init();
        assert!(r.is_err(), "Should return an error");
        assert_eq!(
            r.unwrap_err(),
            "Minimum learning rate must be at least 0 and at most equal to the initial learning \
             rate",
            "Error messages should match",
        );
    }

    #[test]
    fn config_min_lr_too_high() {
        let r = CosineAnnealingLrSchedulerConfig::new(0.5, 10)
            .with_min_lr(0.6)
            .init();
        assert!(r.is_err(), "Should return an error");
        assert_eq!(
            r.unwrap_err(),
            "Minimum learning rate must be at least 0 and at most equal to the initial learning \
             rate",
            "Error messages should match",
        );
    }

    #[test]
    fn config_num_iters_too_low() {
        let r = CosineAnnealingLrSchedulerConfig::new(0.5, 0).init();
        assert!(r.is_err(), "Should return an error");
        assert_eq!(
            r.unwrap_err(),
            "Number of iterations must be at least 1",
            "Error messages should match",
        );
    }

    #[test]
    fn test_lr_change() {
        const INITIAL_LR: LearningRate = 0.5;
        const MIN_LR: LearningRate = 0.1;

        let scheduler = CosineAnnealingLrSchedulerConfig::new(INITIAL_LR, 2)
            .with_min_lr(MIN_LR)
            .init()
            .unwrap();
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
        const NUM_ITERS: usize = 9;
        let scheduler = CosineAnnealingLrSchedulerConfig::new(1.0, NUM_ITERS)
            .init()
            .unwrap();
        test_utils::check_save_load(scheduler, NUM_ITERS / 3 * 2);
    }
}
