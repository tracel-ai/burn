use super::LrScheduler;
use crate as burn;
use crate::{config::Config, LearningRate};
use burn_tensor::backend::Backend;

/// The configuration for creating a Cosine Annealing learning rate scheduler with cold restarts.
///
/// This scheduler starts at a learning rate `initial_lr`, then changes the learning rate by following a cosine function
/// with a period of `num_iters` iterations. After `num_iters` iterations, the learning rate is reset to `initial_lr`.
#[derive(Config)]
pub struct CosineAnnealingLrSchedulerConfig {
    // The initial learning rate.
    initial_lr: LearningRate,
    // The final learning rate.
    #[config(default = 0.0)]
    min_lr: LearningRate,
    // The number of iterations before the learning rate is reset.
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
            previous_lr: self.initial_lr,
            min_lr: self.min_lr,
            max_lr: self.initial_lr,
            num_iters: self.num_iters,
            current_iter: 0,
        }
    }
}

/// A Cosine Annealing learning rate scheduler.
///
/// See [CosineAnnealingLrSchedulerConfig] for more information.
#[derive(Clone, Copy, Debug)]
pub struct CosineAnnealingLrScheduler {
    // The previous iteration's learning rate.
    previous_lr: LearningRate,
    min_lr: LearningRate,
    max_lr: LearningRate,
    num_iters: usize,
    current_iter: usize,
}

impl<B: Backend> LrScheduler<B> for CosineAnnealingLrScheduler {
    type Record = (LearningRate, LearningRate, LearningRate, usize, usize);

    fn step(&mut self) -> LearningRate {
        if self.current_iter < self.num_iters {
            self.current_iter += 1;
        } else {
            self.current_iter = 0;
        }
        self.previous_lr = self.min_lr
            + 0.5
                * (self.max_lr - self.min_lr)
                * (1.0
                    + (self.current_iter as f64 / self.num_iters as f64 * std::f64::consts::PI)
                        .cos());
        self.previous_lr
    }

    fn to_record(&self) -> Self::Record {
        (
            self.previous_lr,
            self.min_lr,
            self.max_lr,
            self.num_iters,
            self.current_iter,
        )
    }

    fn load_record(mut self, record: Self::Record) -> Self {
        (
            self.previous_lr,
            self.min_lr,
            self.max_lr,
            self.num_iters,
            self.current_iter,
        ) = record;
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
        const NUM_ITERS: usize = 10;

        let mut scheduler = CosineAnnealingLrSchedulerConfig::new(INITIAL_LR, NUM_ITERS)
            .with_min_lr(MIN_LR)
            .init();

        let mut previous_lr = INITIAL_LR;

        for _ in 0..NUM_ITERS {
            let lr = LrScheduler::<TestBackend>::step(&mut scheduler);
            assert!(
                lr < previous_lr,
                "Learning rate should decrease with each iteration before reaching the specified number of iterations"
            );
            previous_lr = lr;
        }

        assert_eq!(
            previous_lr, MIN_LR,
            "Learning rate should reach the final learning rate"
        );

        assert_eq!(
            LrScheduler::<TestBackend>::step(&mut scheduler),
            INITIAL_LR,
            "Learning rate should be reset after the specified number of iterations"
        );
    }
}
