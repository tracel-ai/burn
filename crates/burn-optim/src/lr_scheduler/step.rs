use burn_core as burn;

use burn::config::Config;
use burn::tensor::backend::Backend;

use super::{LrScheduler, String};
use crate::LearningRate;

/// The configuration for create a [step learning rate scheduler](StepLrScheduler).
///
/// This scheduler returns the learning rate `initial_lr` from the start, and keeps doing so until
/// the same value has been given for `step_size` times. Then it multiplies the learning rate by
/// `gamma` before repeating the process.
///
/// Gamma values out of range (0.0, 1.0) and non-positive initial learning rates are acceptable, but
/// a warning log will be output for such a value in case of mistyping.
///
/// ## Notes
///
/// The [step](StepLrScheduler::step) method of the scheduler panics if it is called more than
/// `i32::MAX + 1` times.
#[derive(Config, Debug)]
pub struct StepLrSchedulerConfig {
    // The learning rate at the initial step.
    initial_lr: LearningRate,
    // The number of iterations over which the learning rate remains unchanged before the next
    // update.
    step_size: usize,
    /// The factor by which the learning rate is multiplied with each update. Default: 0.1.
    #[config(default = 0.1)]
    gamma: f64,
}

impl StepLrSchedulerConfig {
    /// Initializes a [step learning rate scheduler](StepLrScheduler).
    ///
    /// # Errors
    ///
    /// An error will be returned if `step_size` is 0.
    pub fn init(&self) -> Result<StepLrScheduler, String> {
        if self.step_size == 0 {
            return Err("Step size must be greater than 0".into());
        }

        // Atypical values of `initial_lr` and `gamma` are not rejected because they might be useful
        // in some cases like debugging (e.g., https://datascience.stackexchange.com/q/89518).
        if self.initial_lr <= 0.0 {
            log::warn!(
                "Initial learning rate value of {} is not a positive number. Ignore this warning \
                 if it is intended.",
                self.initial_lr
            );
        }
        if self.gamma <= 0.0 || self.gamma >= 1.0 {
            log::warn!(
                "Gamma value of {} is out of range (0.0, 1.0). Ignore this warning if it is \
                 intended.",
                self.gamma
            );
        }

        Ok(StepLrScheduler {
            init_lr: self.initial_lr,
            step_size: self.step_size,
            gamma: self.gamma,
            iter_idx: -1,
        })
    }
}

/// Step learning rate scheduler.
#[derive(Clone, Debug)]
pub struct StepLrScheduler {
    init_lr: LearningRate,
    step_size: usize,
    gamma: f64,
    // The index of the current iteration.
    // `i32` is used for avoiding truncating the exponent when taking powers of `gamma`.
    iter_idx: i32,
}

impl LrScheduler for StepLrScheduler {
    type Record<B: Backend> = i32;

    fn step(&mut self) -> LearningRate {
        self.iter_idx = self
            .iter_idx
            .checked_add(1)
            .expect("`.step()` should be called no more than `i32::MAX + 1` times");
        // Type casting below causes no truncation, as all the values fall within the ranges.
        self.init_lr
            * self
                .gamma
                .powi((self.iter_idx as usize / self.step_size) as i32)
    }

    fn to_record<B: Backend>(&self) -> Self::Record<B> {
        self.iter_idx
    }

    fn load_record<B: Backend>(mut self, record: Self::Record<B>) -> Self {
        self.iter_idx = record;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_utils;
    use super::*;
    use crate::TestBackend;

    // Warning logs for initial LR and gamma are not tested because there seems no straightforward
    // way to do it.
    //
    // Creating a mock logger that collects logs into `String` for later examination seems a possible
    // solution, but unit tests run in the same process in parallel, where the single logger would
    // be shared by multiple tests, so logs from different tests would be mixed up with no easy way
    // to separate them.
    // Using "--test-threads=1" could prevent mixup, but whether the ability to test logging is
    // worth the slowdown would be a question. Also, using a primitive provided by `std` to
    // synchronize the logger across tests is not an option since we need to support `no-std`.
    // Maybe the mocking approach can be reconsidered after we are given an option to run tests in
    // separate processes like what the issue below is proposing:
    //     https://github.com/rust-lang/rust/issues/47506
    //
    // As a side note, a helper crate exists for the exact purpose:
    //     https://crates.io/crates/testing_logger
    // but the crate has been unmaintained and using it would introduce another dependency.

    #[test]
    fn test_config_step_size_zero() {
        let r = StepLrSchedulerConfig::new(1.0, 0).init();
        assert!(r.is_err(), "Should return an error");
    }

    #[test]
    fn test_config_step_size_nonzero() {
        let r = StepLrSchedulerConfig::new(1.0, 1).init();
        assert!(r.is_ok(), "Should return a success value");
    }

    #[test]
    fn test_config_default_gamma() {
        const INIT_LR: LearningRate = 0.4;
        const STEP_SIZE: usize = 2;

        let mut default = StepLrSchedulerConfig::new(INIT_LR, STEP_SIZE)
            .init()
            .unwrap();
        let mut explicit = StepLrSchedulerConfig::new(INIT_LR, STEP_SIZE)
            .with_gamma(0.1)
            .init()
            .unwrap();
        test_utils::compare_steps(&mut default, &mut explicit, 3 * STEP_SIZE);
    }

    #[test]
    fn test_lr_decreasing() {
        let scheduler = StepLrSchedulerConfig::new(0.5, 3)
            .with_gamma(0.1)
            .init()
            .unwrap();
        let expected_lrs = [0.5, 0.5, 0.5, 0.05, 0.05, 0.05, 0.005, 0.005, 0.005];
        test_utils::check_lr_sequence(scheduler, expected_lrs);
    }

    #[test]
    fn test_lr_increasing() {
        let scheduler = StepLrSchedulerConfig::new(0.1, 2)
            .with_gamma(2.0)
            .init()
            .unwrap();
        let expected_lrs = [0.1, 0.1, 0.2, 0.2, 0.4, 0.4];
        test_utils::check_lr_sequence(scheduler, expected_lrs);
    }

    #[test]
    fn test_lr_unchanging() {
        let scheduler = StepLrSchedulerConfig::new(3.1, 1)
            .with_gamma(1.0)
            .init()
            .unwrap();
        let expected_lrs = [3.1, 3.1, 3.1];
        test_utils::check_lr_sequence(scheduler, expected_lrs);
    }

    #[test]
    fn test_save_and_load() {
        const STEP_SIZE: usize = 10;

        let scheduler = StepLrSchedulerConfig::new(0.007, STEP_SIZE)
            .with_gamma(0.03)
            .init()
            .unwrap();
        test_utils::check_save_load(scheduler, 3 * STEP_SIZE / 2);
    }

    // It's too time consuming to actually run a scheduler `i32::MAX` steps, so an approach that
    // depends on private fields is used to implement the test.
    #[test]
    fn test_number_of_calls_within_limit() {
        // Create a scheduler that has already run `i32::MAX` steps
        let mut scheduler = StepLrSchedulerConfig::new(0.1, 2).init().unwrap();
        scheduler = scheduler.load_record::<TestBackend>(i32::MAX - 1);
        scheduler.step();
    }

    #[test]
    #[should_panic = "i32::MAX"]
    fn test_number_of_calls_over_limit() {
        // Create a scheduler that has already run `i32::MAX` steps
        let mut scheduler = StepLrSchedulerConfig::new(0.1, 2).init().unwrap();
        scheduler = scheduler.load_record::<TestBackend>(i32::MAX - 1);
        scheduler.step();
        scheduler.step();
    }
}
