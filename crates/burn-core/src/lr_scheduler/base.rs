use burn_tensor::backend::Backend;

use crate::{record::Record, LearningRate};

/// Learning rate scheduler defines how the learning rate will evolve during training.
pub trait LrScheduler: Send + Sync {
    /// Scheduler associative type to be used when saving and loading the state.
    type Record<B: Backend>: Record<B>;

    /// Perform the scheduler step, potentially updating its state, and returning the effective
    /// learning rate.
    fn step(&mut self) -> LearningRate;

    /// Get the current state of the scheduler as a [record](Record).
    fn to_record<B: Backend>(&self) -> Self::Record<B>;

    /// Load the state of the scheduler as a [record](Record).
    fn load_record<B: Backend>(self, record: Self::Record<B>) -> Self;
}

#[cfg(test)]
pub(super) mod test_utils {
    use super::*;
    use crate::TestBackend;

    pub fn check_lr_sequence<I, S>(mut scheduler: S, expected_lrs: I)
    where
        I: IntoIterator<Item = LearningRate>,
        S: LrScheduler,
    {
        // Depending on how learning rates are computed by the scheduler, floating-point arithmetic
        // error might exceed f64::EPSILON, so we use a larger epsilon here.
        const LOOSE_EPSILON: f64 = 1e-10;

        expected_lrs
            .into_iter()
            .enumerate()
            .for_each(|(i, expected)| {
                let lr = scheduler.step();
                assert!(
                    (lr - expected).abs() < LOOSE_EPSILON,
                    "Scheduled learning rate {lr} is not approximately equal to the expected value \
                     {expected} at step {i}",
                );
            });
    }

    // save_at_step is the number of steps to run the scheduler before saving and loading back its
    // state.
    pub fn check_save_load<S>(mut scheduler: S, save_at_step: usize)
    where
        S: Clone + LrScheduler,
    {
        let mut truth = scheduler.clone();
        // Consume some steps before saving and loading back
        (0..save_at_step).for_each(|_| {
            truth.step();
            scheduler.step();
        });
        let rec = scheduler.to_record::<TestBackend>();
        scheduler = scheduler.load_record::<TestBackend>(rec);

        // Validate that the scheduler resumes from where it left off.
        (save_at_step..2 * save_at_step).for_each(|i| {
            let expected = truth.step();
            let lr = scheduler.step();
            // The two schedulers run with the exact same settings and code,
            // so the difference, if any, should be small enough to fit in f64::EPSILON.
            assert!(
                (lr - expected).abs() < f64::EPSILON,
                "Scheduled learning rate {lr} is not approximately equal to the expected value \
                 {expected} at step {i}",
            );
        });
    }
}
