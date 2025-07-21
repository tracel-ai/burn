use crate::Dataset;
use rand::{SeedableRng, prelude::SliceRandom, rngs::StdRng};
use std::marker::PhantomData;

/// Shuffled a dataset, consider using [sampler dataset](crate::transform::SamplerDataset) is you
/// want a probability distribution that is computed lazily.
pub struct ShuffledDataset<D, I> {
    dataset: D,
    indices: Vec<usize>,
    input: PhantomData<I>,
}

/// Generates a shuffled vector of indices up to a size.
///
/// # Arguments
///
/// * `size` - The size of the dataset to shuffle.
///
/// # Returns
///
/// A vector of shuffled indices.
#[inline(always)]
pub fn shuffled_indices(size: usize, rng: &mut StdRng) -> Vec<usize> {
    let mut indices = (0..size).collect::<Vec<_>>();
    indices.shuffle(rng);
    indices
}

impl<D, I> ShuffledDataset<D, I>
where
    D: Dataset<I>,
{
    /// Creates a new shuffled dataset.
    pub fn new(dataset: D, rng: &mut StdRng) -> Self {
        let indices = shuffled_indices(dataset.len(), rng);
        Self {
            dataset,
            indices,
            input: PhantomData,
        }
    }

    /// Creates a new shuffled dataset with a fixed seed.
    pub fn with_seed(dataset: D, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        Self::new(dataset, &mut rng)
    }
}

impl<D, I> Dataset<I> for ShuffledDataset<D, I>
where
    D: Dataset<I>,
    I: Clone + Send + Sync,
{
    fn get(&self, index: usize) -> Option<I> {
        let index = self.indices.get(index)?;
        self.dataset.get(*index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FakeDataset;

    #[test]
    fn test_shuffled_dataset() {
        let dataset = FakeDataset::<String>::new(27);
        let source_items = dataset.iter().collect::<Vec<_>>();

        let seed = 42;

        let shuffled = ShuffledDataset::with_seed(dataset, seed);

        let mut rng = StdRng::seed_from_u64(seed);
        let indices = shuffled_indices(source_items.len(), &mut rng);

        assert_eq!(&shuffled.indices, &indices);
        assert_eq!(shuffled.len(), source_items.len());

        let expected_items: Vec<_> = shuffled
            .indices
            .iter()
            .map(|&i| source_items[i].to_string())
            .collect();
        assert_eq!(&shuffled.iter().collect::<Vec<_>>(), &expected_items);
    }
}
