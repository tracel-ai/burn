use crate::Dataset;
use rand::prelude::SliceRandom;
use rand::{Rng, SeedableRng, distr::Uniform, rngs::StdRng, seq::IteratorRandom};
use std::{marker::PhantomData, ops::DerefMut, sync::Mutex};

/// Sample items from a dataset.
///
/// This is an convenient way of modeling a dataset as a probability distribution of a fixed size.
/// You have multiple options to instantiate the dataset sampler.
///
/// * With replacement (Default): This is the most efficient way of using the sampler because no state is
///   required to keep indices that have been selected.
///
/// * Without replacement: This has a similar effect to using a
///   [shuffled dataset](crate::transform::ShuffledDataset), but with more flexibility since you can
///   set the dataset to an arbitrary size. Once every item has been used, a new cycle is
///   created with a new random suffle.
pub struct SamplerDataset<D, I> {
    dataset: D,
    size: usize,
    state: Mutex<SamplerState>,
    input: PhantomData<I>,
}

enum SamplerState {
    WithReplacement(StdRng),
    WithoutReplacement(StdRng, Vec<usize>),
}

impl<D, I> SamplerDataset<D, I>
where
    D: Dataset<I>,
    I: Send + Sync,
{
    /// Creates a new sampler dataset with replacement.
    pub fn new(dataset: D, size: usize) -> Self {
        Self {
            dataset,
            size,
            state: Mutex::new(SamplerState::WithReplacement(StdRng::from_os_rng())),
            input: PhantomData,
        }
    }

    /// Creates a new sampler dataset with replacement.
    pub fn with_replacement(dataset: D, size: usize) -> Self {
        Self::new(dataset, size)
    }

    /// Creates a new sampler dataset without replacement.
    ///
    /// When the sample size is less than or equal to the source dataset size,
    /// data will be sampled without replacement from the source dataset in
    /// a uniformly shuffled order.
    ///
    /// When the sample size is greater than the source dataset size,
    /// the entire source dataset will be exhausted before re-sampling,
    /// for each multiple of the source size.
    pub fn without_replacement(dataset: D, size: usize) -> Self {
        Self {
            dataset,
            size,
            state: Mutex::new(SamplerState::WithoutReplacement(
                StdRng::from_os_rng(),
                Vec::new(),
            )),
            input: PhantomData,
        }
    }

    fn index(&self) -> usize {
        let mut state = self.state.lock().unwrap();

        match state.deref_mut() {
            SamplerState::WithReplacement(rng) => {
                rng.sample(Uniform::new(0, self.dataset.len()).unwrap())
            }
            SamplerState::WithoutReplacement(rng, indices) => {
                if indices.is_empty() {
                    // Refill the state.
                    *indices = (0..self.dataset.len()).choose_multiple(rng, self.dataset.len());

                    // From `choose_multiple` documentation:
                    // > Although the elements are selected randomly, the order of elements in
                    // > the buffer is neither stable nor fully random. If random ordering is
                    // > desired, shuffle the result.
                    //
                    // Without this, for size~=ds.size; the indices will return
                    // essentially in a linear order.
                    indices.shuffle(rng);
                }

                indices.pop().expect("Indices are refilled when empty.")
            }
        }
    }
}

impl<D, I> Dataset<I> for SamplerDataset<D, I>
where
    D: Dataset<I>,
    I: Send + Sync,
{
    fn get(&self, index: usize) -> Option<I> {
        if index >= self.size {
            return None;
        }

        self.dataset.get(self.index())
    }

    fn len(&self) -> usize {
        self.size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FakeDataset;
    use std::collections::HashMap;

    #[test]
    fn sampler_dataset_with_replacement_iter() {
        let factor = 3;
        let len_original = 10;
        let dataset_sampler = SamplerDataset::with_replacement(
            FakeDataset::<String>::new(len_original),
            len_original * factor,
        );
        let mut total = 0;

        for _item in dataset_sampler.iter() {
            total += 1;
        }

        assert_eq!(total, factor * len_original);
    }

    #[test]
    fn sampler_dataset_without_replacement_bucket_test() {
        let factor = 3;
        let len_original = 10;
        let dataset_sampler = SamplerDataset::without_replacement(
            FakeDataset::<String>::new(len_original),
            len_original * factor,
        );
        let mut buckets = HashMap::new();

        for item in dataset_sampler.iter() {
            let count = match buckets.get(&item) {
                Some(count) => count + 1,
                None => 1,
            };

            buckets.insert(item, count);
        }

        let mut total = 0;
        for count in buckets.into_values() {
            assert_eq!(count, factor);
            total += count;
        }
        assert_eq!(total, factor * len_original);
    }

    #[test]
    fn sampler_dataset_without_replacement_uniform_order_test() {
        // This is a reversion test on the indices.shuffle(rng) call in SamplerDataset::index().
        let size = 100;
        let dataset_sampler =
            SamplerDataset::without_replacement(FakeDataset::<i32>::new(size), size);

        let indices: Vec<_> = (0..size).map(|_| dataset_sampler.index()).collect();
        let mean_delta = indices
            .windows(2)
            .map(|pair| pair[1].abs_diff(pair[0]))
            .sum::<usize>() as f64
            / (size - 1) as f64;

        let expected = (size + 2) as f64 / 3.0;

        assert!(
            (mean_delta - expected).abs() <= 0.2 * expected,
            "Sampled indices are not uniformly distributed: mean_delta: {mean_delta}, expected: {expected}"
        );
    }
}
