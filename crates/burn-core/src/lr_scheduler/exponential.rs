use super::LrScheduler;
use crate as burn;
use crate::{config::Config, LearningRate};
use burn_tensor::backend::Backend;

/// The configuration for creating an exponential learning rate scheduler.
///
/// This scheduler starts at a learning rate `initial_lr`, then changes the learning rate by multiplying it by a constant
/// `gamma` at every iteration. At any iteration `i`, the learning rate is given by `initial_lr * gamma^i`.
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
            previous_lr: self.initial_lr,
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

impl<B: Backend> LrScheduler<B> for ExponentialLrScheduler {
    type Record = (LearningRate, f64);

    fn step(&mut self) -> LearningRate {
        self.previous_lr *= self.gamma;
        self.previous_lr
    }

    fn to_record(&self) -> Self::Record {
        (self.previous_lr, self.gamma)
    }

    fn load_record(mut self, record: Self::Record) -> Self {
        (self.previous_lr, self.gamma) = record;
        self
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::TestBackend;

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
        const INITIAL_LR: LearningRate = 0.75;
        const GAMMA: f64 = 0.9;
        const NUM_ITERS: usize = 10;
        const EPSILON: f64 = 1e-10;

        let mut scheduler = ExponentialLrSchedulerConfig::new(INITIAL_LR, GAMMA).init();

        let mut previous_lr = INITIAL_LR;

        for _ in 0..NUM_ITERS {
            let lr = LrScheduler::<TestBackend>::step(&mut scheduler);
            assert!(
                lr < previous_lr,
                "Learning rate should decrease with each iteration before reaching the final learning rate"
            );
            previous_lr = lr;
        }

        let expected = INITIAL_LR * GAMMA.powi(NUM_ITERS as i32);
        assert!(
            (previous_lr - expected).abs() < EPSILON,
            "Learning rate should be close to the expected value after reaching the final learning rate"
        );
    }
}
