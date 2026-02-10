use crate::Dataset;
use crate::transform::{RngSource, SizeConfig};
use rand::prelude::SliceRandom;
use rand::{RngExt, distr::Uniform, rngs::StdRng, seq::IteratorRandom};
use std::{marker::PhantomData, ops::DerefMut, sync::Mutex};

/// Options to configure a [SamplerDataset].
#[derive(Debug, Clone, PartialEq)]
pub struct SamplerDatasetOptions {
    /// The sampling mode.
    pub replace_samples: bool,

    /// The size source of the wrapper relative to the dataset.
    pub size_config: SizeConfig,

    /// The source of the random number generator.
    pub rng_source: RngSource,
}

impl Default for SamplerDatasetOptions {
    fn default() -> Self {
        Self {
            replace_samples: true,
            size_config: SizeConfig::Default,
            rng_source: RngSource::Default,
        }
    }
}

impl<T> From<Option<T>> for SamplerDatasetOptions
where
    T: Into<SamplerDatasetOptions>,
{
    fn from(option: Option<T>) -> Self {
        match option {
            Some(option) => option.into(),
            None => Self::default(),
        }
    }
}

impl From<usize> for SamplerDatasetOptions {
    fn from(size: usize) -> Self {
        Self::default().with_replacement().with_fixed_size(size)
    }
}

impl SamplerDatasetOptions {
    /// Set the replacement mode.
    pub fn with_replace_samples(self, replace_samples: bool) -> Self {
        Self {
            replace_samples,
            ..self
        }
    }

    /// Set the replacement mode to WithReplacement.
    pub fn with_replacement(self) -> Self {
        self.with_replace_samples(true)
    }

    /// Set the replacement mode to WithoutReplacement.
    pub fn without_replacement(self) -> Self {
        self.with_replace_samples(false)
    }

    /// Set the size source.
    pub fn with_size<S>(self, source: S) -> Self
    where
        S: Into<SizeConfig>,
    {
        Self {
            size_config: source.into(),
            ..self
        }
    }

    /// Set the size to the size of the source.
    pub fn with_source_size(self) -> Self {
        self.with_size(SizeConfig::Default)
    }

    /// Set the size to a fixed size.
    pub fn with_fixed_size(self, size: usize) -> Self {
        self.with_size(size)
    }

    /// Set the size to be a multiple of the ration and the source size.
    pub fn with_size_ratio(self, size_ratio: f64) -> Self {
        self.with_size(size_ratio)
    }

    /// Set the `RngSource`.
    pub fn with_rng<R>(self, rng: R) -> Self
    where
        R: Into<RngSource>,
    {
        Self {
            rng_source: rng.into(),
            ..self
        }
    }

    /// Use the system rng.
    pub fn with_system_rng(self) -> Self {
        self.with_rng(RngSource::Default)
    }

    /// Use a rng, built from a seed.
    pub fn with_seed(self, seed: u64) -> Self {
        self.with_rng(seed)
    }
}

/// Sample items from a dataset.
///
/// This is a convenient way of modeling a dataset as a probability distribution of a fixed size.
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
    /// ## Arguments
    ///
    /// * `dataset`: the dataset to wrap.
    /// * `options`: the options to configure the sampler dataset.
    ///
    /// ## Examples
    /// ```rust,ignore
    /// use burn_dataset::transform::{
    ///   SamplerDataset,
    ///   SamplerDatasetOptions,
    /// };
    ///
    /// // Examples below assuming `dataset.len()` = `10`.
    ///
    /// // sample size: 5
    /// // WithReplacement
    /// // rng: StdRng::from_os_rng()
    /// SamplerDataset::new(dataset, 5);
    ///
    /// // sample size: 10 (source)
    /// // WithReplacement
    /// // rng: StdRng::from_os_rng()
    /// SamplerDataset::new(dataset, SamplerDatasetOptions::default());
    ///
    /// // sample size: 15
    /// // WithoutReplacement
    /// // rng: StdRng::seed_from_u64(42)
    /// SamplerDataset::new(
    ///   dataset,
    ///   SamplerDatasetOptions::default()
    ///     .with_size(1.5)
    ///     .without_replacement()
    ///     .with_rng(42),
    /// );
    /// ```
    pub fn new<O>(dataset: D, options: O) -> Self
    where
        O: Into<SamplerDatasetOptions>,
    {
        let options = options.into();
        let size = options.size_config.resolve(dataset.len());
        let rng = options.rng_source.into();
        Self {
            dataset,
            size,
            state: Mutex::new(match options.replace_samples {
                true => SamplerState::WithReplacement(rng),
                false => SamplerState::WithoutReplacement(rng, Vec::with_capacity(size)),
            }),
            input: PhantomData,
        }
    }

    /// Creates a new sampler dataset with replacement.
    ///
    /// # Arguments
    ///
    /// - `dataset`: the dataset to wrap.
    /// - `size`: the effective size of the sampled dataset.
    pub fn with_replacement(dataset: D, size: usize) -> Self {
        Self::new(
            dataset,
            SamplerDatasetOptions::default()
                .with_replacement()
                .with_fixed_size(size),
        )
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
        Self::new(
            dataset,
            SamplerDatasetOptions::default()
                .without_replacement()
                .with_fixed_size(size),
        )
    }

    /// Determines if the sampler is using the "with replacement" strategy.
    ///
    /// # Returns
    /// - `true`: If the sampler is configured to sample with replacement.
    /// - `false`: If the sampler is configured to sample without replacement.
    pub fn is_with_replacement(&self) -> bool {
        match self.state.lock().unwrap().deref_mut() {
            SamplerState::WithReplacement(_) => true,
            SamplerState::WithoutReplacement(_, _) => false,
        }
    }

    fn index(&self) -> usize {
        match self.state.lock().unwrap().deref_mut() {
            SamplerState::WithReplacement(rng) => {
                rng.sample(Uniform::new(0, self.dataset.len()).unwrap())
            }
            SamplerState::WithoutReplacement(rng, indices) => {
                if indices.is_empty() {
                    // Refill the state.
                    let idx_range = 0..self.dataset.len();
                    for _ in 0..(self.size / self.dataset.len()) {
                        // No need to `.choose_multiple` here because we're using
                        // the entire source range; and `.choose_multiple` will
                        // not return a random sample anyway.
                        indices.extend(idx_range.clone())
                    }

                    // From `choose_multiple` documentation:
                    // > Although the elements are selected randomly, the order of elements in
                    // > the buffer is neither stable nor fully random. If random ordering is
                    // > desired, shuffle the result.
                    indices.extend(idx_range.sample(rng, self.size - indices.len()));

                    // The real shuffling is done here.
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
    #![allow(clippy::bool_assert_comparison)]

    use super::*;
    use crate::FakeDataset;
    use rand::SeedableRng;
    use std::collections::HashMap;

    #[test]
    fn test_samplerdataset_options() {
        let options = SamplerDatasetOptions::default();
        assert_eq!(options.replace_samples, true);
        assert_eq!(options.size_config, SizeConfig::Default);
        assert_eq!(options.rng_source, RngSource::Default);

        // ReplacementMode
        let options = options.with_replace_samples(false);
        assert_eq!(options.replace_samples, false);
        let options = options.with_replacement();
        assert_eq!(options.replace_samples, true);
        let options = options.without_replacement();
        assert_eq!(options.replace_samples, false);

        // SourceSize
        let options = options.with_size(SizeConfig::Default);
        assert_eq!(options.size_config, SizeConfig::Default);
        let options = options.with_source_size();
        assert_eq!(options.size_config, SizeConfig::Default);
        let options = options.with_fixed_size(10);
        assert_eq!(options.size_config, SizeConfig::Fixed(10));
        let options = options.with_size_ratio(1.5);
        assert_eq!(options.size_config, SizeConfig::Ratio(1.5));

        // RngSource
        let options = options.with_system_rng();
        assert_eq!(options.rng_source, RngSource::Default);
        let options = options.with_seed(42);
        assert_eq!(options.rng_source, RngSource::Seed(42));
        let mut rng = StdRng::seed_from_u64(9);
        let options = options.with_rng(&mut rng);
        matches!(options.rng_source, RngSource::Rng(_));
    }

    #[test]
    fn sampler_dataset_constructors_test() {
        let ds = SamplerDataset::new(FakeDataset::<u32>::new(10), 15);
        assert_eq!(ds.len(), 15);
        assert_eq!(ds.dataset.len(), 10);
        assert!(ds.is_with_replacement());

        let ds = SamplerDataset::with_replacement(FakeDataset::<u32>::new(10), 15);
        assert_eq!(ds.len(), 15);
        assert_eq!(ds.dataset.len(), 10);
        assert!(ds.is_with_replacement());

        let ds = SamplerDataset::without_replacement(FakeDataset::<u32>::new(10), 15);
        assert_eq!(ds.len(), 15);
        assert_eq!(ds.dataset.len(), 10);
        assert!(!ds.is_with_replacement());
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

        let dataset_sampler = SamplerDataset::new(
            FakeDataset::<String>::new(len_original),
            SamplerDatasetOptions::default()
                .without_replacement()
                .with_size_ratio(factor as f64),
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
        let size = 1000;
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
            (mean_delta - expected).abs() <= 0.25 * expected,
            "Sampled indices are not uniformly distributed: mean_delta: {mean_delta}, expected: {expected}"
        );
    }
}
