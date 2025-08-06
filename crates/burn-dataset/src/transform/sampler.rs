use crate::Dataset;
use rand::prelude::SliceRandom;
use rand::{Rng, SeedableRng, distr::Uniform, rngs::StdRng, seq::IteratorRandom};
use std::cmp::min;
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
    fn new_from_state(dataset: D, size: usize, state: SamplerState) -> Self {
        Self {
            dataset,
            size,
            state: Mutex::new(state),
            input: PhantomData,
        }
    }

    /// Creates a builder for the given dataset.
    pub fn builder(dataset: D) -> SamplerDatasetBuilder<D, I> {
        SamplerDatasetBuilder::for_dataset(dataset)
    }

    /// Creates a new sampler dataset with replacement.
    ///
    /// # Arguments
    ///
    /// - `dataset`: the dataset to wrap.
    /// - `size`: the effective size of the sampled dataset.
    pub fn new(dataset: D, size: usize) -> Self {
        Self::with_replacement(dataset, size)
    }

    /// Creates a new sampler dataset with replacement.
    ///
    /// # Arguments
    ///
    /// - `dataset`: the dataset to wrap.
    /// - `size`: the effective size of the sampled dataset.
    pub fn with_replacement(dataset: D, size: usize) -> Self {
        Self::with_replacement_from_rng(dataset, size, StdRng::from_os_rng())
    }

    /// Creates a new sampler dataset with replacement.
    ///
    /// # Arguments
    ///
    /// - `dataset`: the dataset to wrap.
    /// - `size`: the effective size of the sampled dataset.
    /// - `seed`: the seed to seed the rng with.
    pub fn with_replacement_from_seed(dataset: D, size: usize, seed: u64) -> Self {
        Self::with_replacement_from_rng(dataset, size, StdRng::seed_from_u64(seed))
    }

    /// Creates a new sampler dataset with replacement.
    ///
    /// # Arguments
    ///
    /// - `dataset`: the dataset to wrap.
    /// - `size`: the effective size of the sampled dataset.
    /// - `rng`: the rng to use.
    pub fn with_replacement_from_rng(dataset: D, size: usize, rng: StdRng) -> Self {
        Self::new_from_state(dataset, size, SamplerState::WithReplacement(rng))
    }

    /// Creates a new sampler dataset without replacement.
    ///
    /// When the sample size is less than or equal to the source dataset size,
    /// data will be sampled without replacement from the source dataset in
    /// a uniformly shuffled order.
    ///
    /// When the sample size is greater than the source dataset size,
    /// the entire source dataset will be sampled once for every multiple
    /// of the size ratios; with the remaining samples taken without replacement
    /// uniformly from the source. All samples will be returned uniformly shuffled.
    ///
    /// # Arguments
    /// - `dataset`: the dataset to wrap.
    /// - `size`: the effective size of the sampled dataset.
    pub fn without_replacement(dataset: D, size: usize) -> Self {
        Self::without_replacement_from_rng(dataset, size, StdRng::from_os_rng())
    }

    /// Creates a new sampler dataset without replacement.
    ///
    /// When the sample size is less than or equal to the source dataset size,
    /// data will be sampled without replacement from the source dataset in
    /// a uniformly shuffled order.
    ///
    /// When the sample size is greater than the source dataset size,
    /// the entire source dataset will be sampled once for every multiple
    /// of the size ratios; with the remaining samples taken without replacement
    /// uniformly from the source. All samples will be returned uniformly shuffled.
    ///
    /// # Arguments
    /// - `dataset`: the dataset to wrap.
    /// - `size`: the effective size of the sampled dataset.
    /// - `seed`: the seed to seed the rng with.
    pub fn without_replacement_from_seed(dataset: D, size: usize, seed: u64) -> Self {
        Self::without_replacement_from_rng(dataset, size, StdRng::seed_from_u64(seed))
    }

    /// Creates a new sampler dataset without replacement.
    ///
    /// When the sample size is less than or equal to the source dataset size,
    /// data will be sampled without replacement from the source dataset in
    /// a uniformly shuffled order.
    ///
    /// When the sample size is greater than the source dataset size,
    /// the entire source dataset will be sampled once for every multiple
    /// of the size ratios; with the remaining samples taken without replacement
    /// uniformly from the source. All samples will be returned uniformly shuffled.
    ///
    /// # Arguments
    /// - `dataset`: the dataset to wrap.
    /// - `size`: the effective size of the sampled dataset.
    /// - `rng`: the rng to use.
    pub fn without_replacement_from_rng(dataset: D, size: usize, rng: StdRng) -> Self {
        Self::new_from_state(
            dataset,
            size,
            SamplerState::WithoutReplacement(rng, Vec::with_capacity(size)),
        )
    }

    /// Determines if the sampler is using the "with replacement" strategy.
    ///
    /// # Returns
    /// - `true`: If the sampler is configured to sample with replacement.
    /// - `false`: If the sampler is configured to sample without replacement.
    pub fn uses_replacement(&self) -> bool {
        match self.state.lock().unwrap().deref_mut() {
            SamplerState::WithReplacement(_) => true,
            SamplerState::WithoutReplacement(_, _) => false,
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
                    while indices.len() < self.size {
                        let k = min(self.size - indices.len(), self.dataset.len());
                        indices.extend((0..self.dataset.len()).choose_multiple(rng, k));
                    }

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

/// Fluent builder for SamplerDataset.
pub struct SamplerDatasetBuilder<D, I>
where
    D: Dataset<I>,
    I: Send + Sync,
{
    dataset: D,
    size: Option<usize>,
    rng: Option<StdRng>,
    with_replacement: bool,
    input: PhantomData<I>,
}

impl<D, I> SamplerDatasetBuilder<D, I>
where
    D: Dataset<I>,
    I: Send + Sync,
{
    /// Create a new builder for the dataset.
    ///
    /// Defaults to:
    /// - Using replacement.
    /// - Using the size of the dataset.
    /// - Using a system rng.
    ///
    /// # Arguments
    ///
    /// - `dataset`: the dataset to wrap.
    pub fn for_dataset(dataset: D) -> Self {
        Self {
            dataset,
            size: None,
            rng: None,
            with_replacement: true,
            input: PhantomData,
        }
    }

    /// Set a size for the sample dataset.
    pub fn with_size(self, size: usize) -> Self {
        Self {
            size: Some(size),
            ..self
        }
    }

    /// Sets the size as a ratio of the wrapped dataset size.
    pub fn with_size_ratio(self, size_ratio: f64) -> Self {
        assert!(
            size_ratio > 0.0,
            "size_ratio must be positive: {size_ratio}"
        );
        let size = ((self.dataset.len() as f64) * size_ratio) as usize;
        self.with_size(size)
    }

    /// Sets a seed for constructing the rng.
    pub fn with_seed(self, seed: u64) -> Self {
        Self {
            rng: Some(StdRng::seed_from_u64(seed)),
            ..self
        }
    }

    /// Sets the rng to use.
    pub fn with_rng(self, rng: StdRng) -> Self {
        Self {
            rng: Some(rng),
            ..self
        }
    }

    /// Enables/Disables replacement sampling.
    pub fn with_replacement(self, replacement: bool) -> Self {
        Self {
            with_replacement: replacement,
            ..self
        }
    }

    /// Build the SamplerDataset.
    pub fn build(self) -> SamplerDataset<D, I> {
        let size = self.size.unwrap_or(self.dataset.len());
        let rng = self.rng.unwrap_or_else(StdRng::from_os_rng);
        if self.with_replacement {
            SamplerDataset::with_replacement_from_rng(self.dataset, size, rng)
        } else {
            SamplerDataset::without_replacement_from_rng(self.dataset, size, rng)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FakeDataset;
    use std::collections::HashMap;

    #[test]
    fn sampler_dataset_builder() {
        let ds = SamplerDataset::builder(FakeDataset::<u32>::new(10))
            .with_size(15)
            .with_replacement(false)
            .with_seed(42)
            .build();
        assert_eq!(ds.len(), 15);
        assert_eq!(ds.dataset.len(), 10);
        assert!(!ds.uses_replacement());

        let ds = SamplerDataset::builder(FakeDataset::<u32>::new(10))
            .with_size_ratio(1.5)
            .with_replacement(true)
            .build();
        assert_eq!(ds.len(), 15);
        assert_eq!(ds.dataset.len(), 10);
        assert!(ds.uses_replacement());
    }

    #[test]
    fn sampler_dataset_constructors_test() {
        let ds = SamplerDataset::new(FakeDataset::<u32>::new(10), 15);
        assert_eq!(ds.len(), 15);
        assert_eq!(ds.dataset.len(), 10);
        assert!(ds.uses_replacement());

        let ds = SamplerDataset::with_replacement(FakeDataset::<u32>::new(10), 15);
        assert_eq!(ds.len(), 15);
        assert_eq!(ds.dataset.len(), 10);
        assert!(ds.uses_replacement());

        let ds = SamplerDataset::with_replacement_from_seed(FakeDataset::<u32>::new(10), 15, 42);
        assert_eq!(ds.len(), 15);
        assert_eq!(ds.dataset.len(), 10);
        assert!(ds.uses_replacement());

        let ds = SamplerDataset::with_replacement_from_rng(
            FakeDataset::<u32>::new(10),
            15,
            StdRng::seed_from_u64(42),
        );
        assert_eq!(ds.len(), 15);
        assert_eq!(ds.dataset.len(), 10);
        assert!(ds.uses_replacement());

        let ds = SamplerDataset::without_replacement(FakeDataset::<u32>::new(10), 15);
        assert_eq!(ds.len(), 15);
        assert_eq!(ds.dataset.len(), 10);
        assert!(!ds.uses_replacement());

        let ds = SamplerDataset::without_replacement_from_seed(FakeDataset::<u32>::new(10), 15, 42);
        assert_eq!(ds.len(), 15);
        assert_eq!(ds.dataset.len(), 10);
        assert!(!ds.uses_replacement());

        let ds = SamplerDataset::without_replacement_from_rng(
            FakeDataset::<u32>::new(10),
            15,
            StdRng::seed_from_u64(42),
        );
        assert_eq!(ds.len(), 15);
        assert_eq!(ds.dataset.len(), 10);
        assert!(!ds.uses_replacement());
    }

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
