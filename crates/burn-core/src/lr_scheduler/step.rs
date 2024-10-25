use burn_tensor::backend::Backend;

use crate as burn;

use super::LrScheduler;
use crate::{config::Config, LearningRate};

/// The configuration for create a [step learning rate scheduler](StepLrScheduler).
///
/// This scheduler returns the learning rate `initial_lr` from the start, and keeps doing so until
/// the same value has been given for `step_size` times. Then it multiplies the learning rate by
/// `gamma` before repeating the process.
///
/// ## Notes
///
/// The [step](StepLrScheduler::step) method of the scheduler panics if it is called more than
/// `i32::MAX + 1` times.
#[derive(Config)]
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
    /// An error will be returned if any field is out of the acceptable range.
    pub fn init(&self) -> Result<StepLrScheduler, String> {
        // `initial_lr` and `gamma` are not checked because atypical values such as zero and
        // negative values might be useful in some cases like debugging (e.g.,
        // https://datascience.stackexchange.com/q/89518).
        if self.step_size == 0 {
            return Err("Step size must be greater than 0".into());
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

    #[test]
    fn test_config_step_size_zero() {
        assert!(StepLrSchedulerConfig::new(1.0, 0).init().is_err());
    }

    #[test]
    fn test_config_step_size_nonzero() {
        assert!(StepLrSchedulerConfig::new(1.0, 1).init().is_ok());
    }

    #[test]
    fn test_config_default_gamma() {
        const INIT_LR: LearningRate = 0.4;
        const STEP_SIZE: usize = 2;

        let mut default = create_scheduler_unchecked(INIT_LR, STEP_SIZE, None);
        let mut explicit = create_scheduler_unchecked(INIT_LR, STEP_SIZE, Some(0.1));
        test_utils::compare_steps(&mut default, &mut explicit, 3 * STEP_SIZE);
    }

    #[test]
    fn test_lr_decreasing() {
        let scheduler = create_scheduler_unchecked(0.5, 3, Some(0.1));
        let expected_lrs = [0.5, 0.5, 0.5, 0.05, 0.05, 0.05, 0.005, 0.005, 0.005];
        test_utils::check_lr_sequence(scheduler, expected_lrs);
    }

    #[test]
    fn test_lr_increasing() {
        let scheduler = create_scheduler_unchecked(0.1, 2, Some(2.0));
        let expected_lrs = [0.1, 0.1, 0.2, 0.2, 0.4, 0.4];
        test_utils::check_lr_sequence(scheduler, expected_lrs);
    }

    #[test]
    fn test_lr_unchanging() {
        let scheduler = create_scheduler_unchecked(3.1, 1, Some(1.0));
        let expected_lrs = [3.1, 3.1, 3.1];
        test_utils::check_lr_sequence(scheduler, expected_lrs);
    }

    #[test]
    fn test_save_and_load() {
        const STEP_SIZE: usize = 10;

        let scheduler = create_scheduler_unchecked(0.007, STEP_SIZE, Some(0.03));
        test_utils::check_save_load(scheduler, 3 * STEP_SIZE / 2);
    }

    // It's too time consuming to actually run a scheduler `i32::MAX` steps, so an approach that
    // depends on private fields is used to implement the test.
    #[test]
    fn test_number_of_calls_within_limit() {
        // Create a scheduler that has already run `i32::MAX` steps
        let mut scheduler = create_scheduler_unchecked(0.1, 2, None);
        scheduler = scheduler.load_record::<TestBackend>(i32::MAX - 1);
        scheduler.step();
    }

    #[test]
    #[should_panic = "i32::MAX"]
    fn test_number_of_calls_over_limit() {
        // Create a scheduler that has already run `i32::MAX` steps
        let mut scheduler = create_scheduler_unchecked(0.1, 2, None);
        scheduler = scheduler.load_record::<TestBackend>(i32::MAX - 1);
        scheduler.step();
        scheduler.step();
    }

    // Create a scheduler with valid parameters (so no boilerplate code for error handling needed).
    fn create_scheduler_unchecked(
        init_lr: LearningRate,
        step_size: usize,
        gamma: Option<f64>,
    ) -> StepLrScheduler {
        let mut config = StepLrSchedulerConfig::new(init_lr, step_size);
        if let Some(g) = gamma {
            config = config.with_gamma(g);
        }
        config
            .init()
            .expect("A scheduler should be created successfully")
    }
}
