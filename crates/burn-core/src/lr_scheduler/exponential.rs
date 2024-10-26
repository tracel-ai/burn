use super::LrScheduler;
use crate as burn;
use crate::{config::Config, LearningRate};
use burn_tensor::backend::Backend;

/// The configuration for creating an [exponential learning rate scheduler](ExponentialLrScheduler).
///
/// This scheduler returns the learning rate `initial_lr` at the first step, then multiplies it by
/// a constant `gamma` at every iteration. At any iteration `i` (which starts from 0), the learning
/// rate is given by `initial_lr * gamma^i`.
#[derive(Config)]
pub struct ExponentialLrSchedulerConfig {
    // The initial learning rate.
    initial_lr: LearningRate,
    // The constant that the learning rate is multiplied by on each iteration.
    gamma: f64,
}

impl ExponentialLrSchedulerConfig {
    /// Initializes a [exponential learning rate scheduler](ExponentialLrScheduler).
    ///
    /// # Panics
    /// This function panics if `initial_lr` and `gamma` are not between 0 and 1.
    pub fn init(&self) -> ExponentialLrScheduler {
        assert!(
            self.initial_lr > 0. && self.initial_lr <= 1.,
            "Initial learning rate must be greater than 0 and at most 1"
        );
        assert!(
            self.gamma > 0. && self.gamma <= 1.,
            "Gamma must be greater than 0 and at most 1"
        );

        ExponentialLrScheduler {
            // Such an initial value eliminates the need for special-case handling of the first
            // learning rate.
            previous_lr: self.initial_lr / self.gamma,
            gamma: self.gamma,
        }
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
    #[should_panic = "Initial learning rate must be greater than 0 and at most 1"]
    fn config_initial_lr_too_low() {
        ExponentialLrSchedulerConfig::new(0., 0.5).init();
    }

    #[test]
    #[should_panic = "Initial learning rate must be greater than 0 and at most 1"]
    fn config_initial_lr_too_high() {
        ExponentialLrSchedulerConfig::new(1.5, 0.5).init();
    }

    #[test]
    #[should_panic = "Gamma must be greater than 0 and at most 1"]
    fn config_gamma_too_low() {
        ExponentialLrSchedulerConfig::new(0.5, 0.0).init();
    }

    #[test]
    #[should_panic = "Gamma must be greater than 0 and at most 1"]
    fn config_gamma_too_high() {
        ExponentialLrSchedulerConfig::new(0.5, 1.5).init();
    }

    #[test]
    fn test_lr_change() {
        const INITIAL_LR: LearningRate = 0.8;
        const GAMMA: f64 = 0.1;

        let scheduler = ExponentialLrSchedulerConfig::new(INITIAL_LR, GAMMA).init();
        let expected_lrs = [0.8, 0.08, 0.008, 0.0008, 0.00008];
        test_utils::check_lr_sequence(scheduler, expected_lrs);
    }

    #[test]
    fn test_save_and_load() {
        const INITIAL_LR: LearningRate = 0.083;
        const GAMMA: f64 = 0.3;
        let scheduler = ExponentialLrSchedulerConfig::new(INITIAL_LR, GAMMA).init();
        test_utils::check_save_load(scheduler, 7);
    }
}
