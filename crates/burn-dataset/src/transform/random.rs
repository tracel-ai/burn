use crate::Dataset;
use crate::transform::SelectionDataset;
use rand::rngs::StdRng;

/// A Shuffled a dataset.
///
/// This is a thin wrapper around a [SelectionDataset] which selects and shuffles
/// the full indices of the original dataset.
///
/// Consider using [sampler dataset](crate::transform::SamplerDataset) if you
/// want a probability distribution which is computed lazily.
pub struct ShuffledDataset<D, I> {
    wrapped: SelectionDataset<D, I>,
}

impl<D, I> ShuffledDataset<D, I>
where
    D: Dataset<I>,
{
    /// Creates a new selection dataset with shuffled indices.
    ///
    /// Selects every index of the dataset and shuffles them
    /// using `shuffled_indices`.
    ///
    /// # Arguments
    ///
    /// * `dataset` - The original dataset to select from.
    /// * `rng` - A mutable reference to a random number generator.
    ///
    /// # Returns
    ///
    /// A new `SelectionDataset` with shuffled indices.
    pub fn new(dataset: D, rng: &mut StdRng) -> Self {
        Self {
            wrapped: SelectionDataset::new_shuffled(dataset, rng),
        }
    }

    /// Creates a new selection dataset with shuffled indices using a fixed seed.
    ///
    /// Selects every index of the dataset and shuffles them
    /// using `shuffled_indices`; seeds the random number generator with the provided seed.
    ///
    /// # Arguments
    ///
    /// * `dataset` - The original dataset to select from.
    /// * `seed` - A fixed seed for the random number generator.
    ///
    /// # Returns
    ///
    /// A new `SelectionDataset` with shuffled indices.
    pub fn with_seed(dataset: D, seed: u64) -> Self {
        Self {
            wrapped: SelectionDataset::new_shuffled_with_seed(dataset, seed),
        }
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

    #[test]
    fn test_shuffled_dataset() {
        let dataset = FakeDataset::<String>::new(27);
        let source_items = dataset.iter().collect::<Vec<_>>();

        let seed = 42;

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
