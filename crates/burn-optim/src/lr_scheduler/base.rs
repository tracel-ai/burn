pub(super) use alloc::string::String;
use burn_core as burn;

use burn::record::Record;
use burn::tensor::backend::Backend;

use crate::LearningRate;

/// Learning rate scheduler defines how the learning rate will evolve during training.
pub trait LrScheduler: Clone + Send + Sync {
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

    // A small tolerance for learning rate comparisons. Depending on how learning rates are
    // computed, floating-point arithmetic error might exceed f64::EPSILON, so a larger value is
    // used here.
    const LOOSE_EPSILON: LearningRate = 1e-10;

    pub fn check_lr_sequence<I, S>(mut scheduler: S, expected_lrs: I)
    where
        I: IntoIterator<Item = LearningRate>,
        S: LrScheduler,
    {
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
        compare_steps(&mut scheduler, &mut truth, save_at_step);
    }

    // Check if two schedulers produce the same learning rate sequences over the specified number of
    // steps.
    pub fn compare_steps<S: LrScheduler>(a: &mut S, b: &mut S, num_steps: usize) {
        (0..num_steps).for_each(|i| {
            let lr_a = a.step();
            let lr_b = b.step();
            assert!(
                (lr_a - lr_b).abs() < LOOSE_EPSILON,
                "The two learning rates ({lr_a}, {lr_b}) at position {i} in the remaining \
                 sequences are not approximately equal",
            );
        });
    }
}
