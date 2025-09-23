use burn_core as burn;

use super::{LrScheduler, String};
use crate::LearningRate;
use burn::config::Config;
use burn::tensor::backend::Backend;

/// The configuration for creating an [exponential learning rate scheduler](ExponentialLrScheduler).
///
/// This scheduler returns the learning rate `initial_lr` at the first step, then multiplies it by
/// a constant `gamma` at every iteration. At any iteration `i` (which starts from 0), the learning
/// rate is given by `initial_lr * gamma^i`.
#[derive(Config, Debug)]
pub struct ExponentialLrSchedulerConfig {
    // The initial learning rate.
    initial_lr: LearningRate,
    // The constant that the learning rate is multiplied by on each iteration.
    gamma: f64,
}

impl ExponentialLrSchedulerConfig {
    /// Initializes a [exponential learning rate scheduler](ExponentialLrScheduler).
    ///
    /// # Errors
    ///
    /// An error will be returned if any of the following conditions is true:
    ///
    /// * `initial_lr` is out of range (0.0, 1.0]
    /// * `gamma` is out of range (0.0, 1.0]
    pub fn init(&self) -> Result<ExponentialLrScheduler, String> {
        if self.initial_lr <= 0. || self.initial_lr > 1. {
            return Err("Initial learning rate must be greater than 0 and at most 1".into());
        }
        if self.gamma <= 0. || self.gamma > 1. {
            return Err("Gamma must be greater than 0 and at most 1".into());
        }

        Ok(ExponentialLrScheduler {
            // Such an initial value eliminates the need for special-case handling of the first
            // learning rate.
            previous_lr: self.initial_lr / self.gamma,
            gamma: self.gamma,
        })
    }
}

/// A exponential learning rate scheduler.
///
/// See [ExponentialLrSchedulerConfig] for more information.
#[derive(Clone, Copy, Debug)]
pub struct ExponentialLrScheduler {
    // The previous iteration's learning rate.
    previous_lr: LearningRate,
    // The constant that the learning rate is multiplied by on each iteration.
    gamma: f64,
}

impl LrScheduler for ExponentialLrScheduler {
    type Record<B: Backend> = LearningRate;

    fn step(&mut self) -> LearningRate {
        self.previous_lr *= self.gamma;
        self.previous_lr
    }

    fn to_record<B: Backend>(&self) -> Self::Record<B> {
        self.previous_lr
    }

    fn load_record<B: Backend>(mut self, record: Self::Record<B>) -> Self {
        self.previous_lr = record;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_utils;
    use super::*;

    #[test]
    fn config_initial_lr_too_low() {
        let r = ExponentialLrSchedulerConfig::new(0., 0.5).init();
        assert!(r.is_err(), "Should return an error");
        assert_eq!(
            r.unwrap_err(),
            "Initial learning rate must be greater than 0 and at most 1",
            "Error messages should match",
        );
    }

    #[test]
    fn config_initial_lr_too_high() {
        let r = ExponentialLrSchedulerConfig::new(1.5, 0.5).init();
        assert!(r.is_err(), "Should return an error");
        assert_eq!(
            r.unwrap_err(),
            "Initial learning rate must be greater than 0 and at most 1",
            "Error messages should match",
        );
    }

    #[test]
    fn config_gamma_too_low() {
        let r = ExponentialLrSchedulerConfig::new(0.5, 0.0).init();
        assert!(r.is_err(), "Should return an error");
        assert_eq!(
            r.unwrap_err(),
            "Gamma must be greater than 0 and at most 1",
            "Error messages should match",
        );
    }

    #[test]
    fn config_gamma_too_high() {
        let r = ExponentialLrSchedulerConfig::new(0.5, 1.5).init();
        assert!(r.is_err(), "Should return an error");
        assert_eq!(
            r.unwrap_err(),
            "Gamma must be greater than 0 and at most 1",
            "Error messages should match",
        );
    }

    #[test]
    fn test_lr_change() {
        let scheduler = ExponentialLrSchedulerConfig::new(0.8, 0.1).init().unwrap();
        let expected_lrs = [0.8, 0.08, 0.008, 0.0008, 0.00008];
        test_utils::check_lr_sequence(scheduler, expected_lrs);
    }

    #[test]
    fn test_save_and_load() {
        let scheduler = ExponentialLrSchedulerConfig::new(0.083, 0.3)
            .init()
            .unwrap();
        test_utils::check_save_load(scheduler, 7);
    }
}
