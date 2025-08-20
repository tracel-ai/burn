use crate::Dataset;
use crate::transform::{RngSource, SelectionDataset};

/// A Shuffled a dataset.
///
/// This is a thin wrapper around a [SelectionDataset] which selects and shuffles
/// the full indices of the original dataset.
///
/// Consider using [SelectionDataset] if you are only interested in
/// shuffling mechanisms.
///
/// Consider using [sampler dataset](crate::transform::SamplerDataset) if you
/// want a probability distribution which is computed lazily.
pub struct ShuffledDataset<D, I>
where
    D: Dataset<I>,
    I: Clone + Send + Sync,
{
    wrapped: SelectionDataset<D, I>,
}

impl<D, I> ShuffledDataset<D, I>
where
    D: Dataset<I>,
    I: Clone + Send + Sync,
{
    /// Creates a new selection dataset with shuffled indices.
    ///
    /// This is a thin wrapper around `SelectionDataset::new_shuffled`.
    ///
    /// # Arguments
    ///
    /// * `dataset` - The original dataset to select from.
    /// * `rng_source` - The source of the random number generator.
    ///
    /// # Returns
    ///
    /// A new `ShuffledDataset`.
    pub fn new<R>(dataset: D, rng_source: R) -> Self
    where
        R: Into<RngSource>,
    {
        Self {
            wrapped: SelectionDataset::new_shuffled(dataset, rng_source),
        }
    }

    /// Creates a new selection dataset with shuffled indices using a fixed seed.
    ///
    /// This is a thin wrapper around `SelectionDataset::new_shuffled_with_seed`.
    ///
    /// # Arguments
    ///
    /// * `dataset` - The original dataset to select from.
    /// * `seed` - A fixed seed for the random number generator.
    ///
    /// # Returns
    ///
    /// A new `ShuffledDataset`.
    #[deprecated(since = "0.19.0", note = "Use `new(dataset, seed)` instead`")]
    pub fn with_seed(dataset: D, seed: u64) -> Self {
        Self::new(dataset, seed)
    }
}

impl<D, I> Dataset<I> for ShuffledDataset<D, I>
where
    D: Dataset<I>,
    I: Clone + Send + Sync,
{
    fn get(&self, index: usize) -> Option<I> {
        self.wrapped.get(index)
    }

    fn len(&self) -> usize {
        self.wrapped.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FakeDataset;
    use crate::transform::selection::shuffled_indices;
    use rand::SeedableRng;
    use rand::prelude::StdRng;

    #[test]
    fn test_shuffled_dataset() {
        let dataset = FakeDataset::<String>::new(27);
        let source_items = dataset.iter().collect::<Vec<_>>();

        let seed = 42;

        #[allow(deprecated)]
        let shuffled = ShuffledDataset::with_seed(dataset, seed);

        let mut rng = StdRng::seed_from_u64(seed);
        let indices = shuffled_indices(source_items.len(), &mut rng);

        assert_eq!(shuffled.len(), source_items.len());

        let expected_items: Vec<_> = indices
            .iter()
            .map(|&i| source_items[i].to_string())
            .collect();
        assert_eq!(&shuffled.iter().collect::<Vec<_>>(), &expected_items);
    }
}
