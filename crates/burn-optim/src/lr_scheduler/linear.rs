use burn_core as burn;

use super::{LrScheduler, String};
use crate::LearningRate;
use burn::config::Config;
use burn::tensor::backend::Backend;

/// The configuration for creating a [linear learning rate scheduler](LinearLrScheduler).
///
/// This scheduler returns the learning rate `initial_lr` at the first step, then changes it by a
/// constant amount on each iteration until reaching a final learning rate `final_lr`. The
/// `num_iters` parameter controls how many iterations are needed to go from `initial_lr` to
/// `final_lr`.
#[derive(Config, Debug)]
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
    /// # Errors
    ///
    /// An error will be returned if any of the following conditions is true:
    ///
    /// * `initial_lr` is out of range (0.0, 1.0]
    /// * `final_lr` is out of range [0.0, 1.0]
    /// * `num_iters` is 0
    pub fn init(&self) -> Result<LinearLrScheduler, String> {
        if self.initial_lr <= 0. || self.initial_lr > 1. {
            return Err("Initial learning rate must be greater than 0 and at most 1".into());
        }
        if self.final_lr < 0. || self.final_lr > 1. {
            return Err("Final learning rate must be at least 0 and at most 1".into());
        }
        if self.num_iters == 0 {
            return Err("Number of iterations must be at least 1".into());
        }

        Ok(LinearLrScheduler {
            final_lr: self.final_lr,
            step_size: (self.final_lr - self.initial_lr) / self.num_iters as f64,
            remaining_iters: self.num_iters + 1,
        })
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
    fn config_initial_lr_too_low() {
        let r = LinearLrSchedulerConfig::new(0., 0.5, 100).init();
        assert!(r.is_err(), "Should return an error");
        assert_eq!(
            r.unwrap_err(),
            "Initial learning rate must be greater than 0 and at most 1",
            "Error messages should match",
        );
    }

    #[test]
    fn config_initial_lr_too_high() {
        let r = LinearLrSchedulerConfig::new(1.5, 0.5, 100).init();
        assert!(r.is_err(), "Should return an error");
        assert_eq!(
            r.unwrap_err(),
            "Initial learning rate must be greater than 0 and at most 1",
            "Error messages should match",
        );
    }

    #[test]
    fn config_final_lr_too_low() {
        let r = LinearLrSchedulerConfig::new(0.5, -0.5, 100).init();
        assert!(r.is_err(), "Should return an error");
        assert_eq!(
            r.unwrap_err(),
            "Final learning rate must be at least 0 and at most 1",
            "Error messages should match",
        );
    }

    #[test]
    fn config_final_lr_too_high() {
        let r = LinearLrSchedulerConfig::new(0.5, 1.5, 100).init();
        assert!(r.is_err(), "Should return an error");
        assert_eq!(
            r.unwrap_err(),
            "Final learning rate must be at least 0 and at most 1",
            "Error messages should match",
        );
    }

    #[test]
    fn config_num_iters_too_low() {
        let r = LinearLrSchedulerConfig::new(0.9, 0.1, 0).init();
        assert!(r.is_err(), "Should return an error");
        assert_eq!(
            r.unwrap_err(),
            "Number of iterations must be at least 1",
            "Error messages should match",
        );
    }

    #[test]
    fn test_lr_decreasing() {
        let scheduler = LinearLrSchedulerConfig::new(0.9, 0.5, 4).init().unwrap();
        let expected_lrs = [0.9, 0.8, 0.7, 0.6, 0.5, 0.5];
        test_utils::check_lr_sequence(scheduler, expected_lrs);
    }

    #[test]
    fn test_lr_increasing() {
        let scheduler = LinearLrSchedulerConfig::new(0.01, 0.04, 3).init().unwrap();
        let expected_lrs = [0.01, 0.02, 0.03, 0.04, 0.04];
        test_utils::check_lr_sequence(scheduler, expected_lrs);
    }

    #[test]
    fn test_lr_unchanging() {
        let scheduler = LinearLrSchedulerConfig::new(0.3, 0.3, 2).init().unwrap();
        let expected_lrs = [0.3, 0.3, 0.3, 0.3];
        test_utils::check_lr_sequence(scheduler, expected_lrs);
    }

    #[test]
    fn test_save_and_load() {
        const NUM_ITERS: usize = 6;
        let scheduler = LinearLrSchedulerConfig::new(1.0, 0.01, NUM_ITERS)
            .init()
            .unwrap();
        test_utils::check_save_load(scheduler, NUM_ITERS / 3 * 2);
    }
}
