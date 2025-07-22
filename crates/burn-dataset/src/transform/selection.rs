use crate::Dataset;
use rand::SeedableRng;
use rand::prelude::SliceRandom;
use rand::rngs::StdRng;
use std::marker::PhantomData;

/// A dataset that selects a subset of indices from an existing dataset.
///
/// Indices may appear multiple times, but they must be within the bounds of the original dataset.
pub struct SelectionDataset<D, I> {
    dataset: D,
    indices: Vec<usize>,
    input: PhantomData<I>,
}

impl<D, I> SelectionDataset<D, I>
where
    D: Dataset<I>,
{
    /// Creates a new selection dataset with the given dataset and indices.
    ///
    /// # Arguments
    ///
    /// * `dataset` - The original dataset to select from.
    /// * `indices` - A slice of indices to select from the dataset.
    ///   These indices must be within the bounds of the dataset.
    ///
    /// # Panics
    ///
    /// Panics if any index is out of bounds for the dataset.
    pub fn new(dataset: D, indices: Vec<usize>) -> Self {
        let size = dataset.len();

        if let Some(idx) = indices.iter().find(|&i| *i >= size) {
            panic!("Index out of bounds for wrapped dataset size: {idx} >= {size}");
        }

        Self {
            dataset,
            indices,
            input: PhantomData,
        }
    }

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
    pub fn shuffled(dataset: D, rng: &mut StdRng) -> Self {
        let indices = shuffled_indices(dataset.len(), rng);
        Self::new(dataset, indices)
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
    pub fn shuffled_with_seed(dataset: D, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        Self::shuffled(dataset, &mut rng)
    }
}

impl<D, I> Dataset<I> for SelectionDataset<D, I>
where
    D: Dataset<I>,
    I: Clone + Send + Sync,
{
    fn get(&self, index: usize) -> Option<I> {
        let index = self.indices.get(index)?;
        self.dataset.get(*index)
    }

    fn len(&self) -> usize {
        self.indices.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FakeDataset;

    #[test]
    fn test_selection_dataset() {
        let source_dataset = FakeDataset::<String>::new(27);

        let indices: Vec<usize> = vec![15, 1, 12, 12];
        let expected: Vec<String> = indices
            .iter()
            .map(|i| source_dataset.get(*i).unwrap())
            .collect();

        let selection = SelectionDataset::new(source_dataset, indices.clone());

        assert_eq!(selection.len(), indices.len());

        let items = selection.iter().collect::<Vec<_>>();

        assert_eq!(items, expected);
    }

    #[test]
    fn test_shuffled_dataset() {
        let dataset = FakeDataset::<String>::new(27);
        let source_items = dataset.iter().collect::<Vec<_>>();

        let seed = 42;

        let shuffled = SelectionDataset::shuffled_with_seed(dataset, seed);

        let mut rng = StdRng::seed_from_u64(seed);
        let indices = shuffled_indices(source_items.len(), &mut rng);

        assert_eq!(&shuffled.indices, &indices);
        assert_eq!(shuffled.len(), source_items.len());

        let expected_items: Vec<_> = indices
            .iter()
            .map(|&i| source_items[i].to_string())
            .collect();
        assert_eq!(&shuffled.iter().collect::<Vec<_>>(), &expected_items);
    }
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
