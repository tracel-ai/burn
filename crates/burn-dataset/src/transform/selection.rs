use crate::Dataset;
use crate::transform::RngSource;
use rand::prelude::SliceRandom;
use rand::rngs::StdRng;
use std::marker::PhantomData;
use std::sync::Arc;

/// Generates a vector of indices from 0 to size - 1.
///
/// # Arguments
///
/// * `size` - The size of the dataset.
///
/// # Returns
///
/// A vector containing indices from 0 to size - 1.
#[inline(always)]
pub fn iota(size: usize) -> Vec<usize> {
    (0..size).collect()
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
    let mut indices = iota(size);
    indices.shuffle(rng);
    indices
}

/// A dataset that selects a subset of indices from an existing dataset.
///
/// Indices may appear multiple times, but they must be within the bounds of the original dataset.
#[derive(Clone)]
pub struct SelectionDataset<D, I>
where
    D: Dataset<I>,
    I: Clone + Send + Sync,
{
    /// The wrapped dataset from which to select indices.
    pub wrapped: Arc<D>,

    /// The indices to select from the wrapped dataset.
    pub indices: Vec<usize>,

    input: PhantomData<I>,
}

impl<D, I> SelectionDataset<D, I>
where
    D: Dataset<I>,
    I: Clone + Send + Sync,
{
    /// Creates a new selection dataset with the given dataset and indices.
    ///
    /// Checks that all indices are within the bounds of the dataset.
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
    pub fn from_indices_checked<S>(dataset: S, indices: Vec<usize>) -> Self
    where
        S: Into<Arc<D>>,
    {
        let dataset = dataset.into();

        let size = dataset.len();
        if let Some(idx) = indices.iter().find(|&i| *i >= size) {
            panic!("Index out of bounds for wrapped dataset size: {idx} >= {size}");
        }

        Self::from_indices_unchecked(dataset, indices)
    }

    /// Creates a new selection dataset with the given dataset and indices without checking bounds.
    ///
    /// # Arguments
    ///
    /// * `dataset` - The original dataset to select from.
    /// * `indices` - A vector of indices to select from the dataset.
    ///
    /// # Safety
    ///
    /// This function does not check if the indices are within the bounds of the dataset.
    pub fn from_indices_unchecked<S>(dataset: S, indices: Vec<usize>) -> Self
    where
        S: Into<Arc<D>>,
    {
        Self {
            wrapped: dataset.into(),
            indices,
            input: PhantomData,
        }
    }

    /// Creates a new selection dataset that selects all indices from the dataset.
    ///
    /// This allocates a 1-to-1 mapping of indices to the dataset size,
    /// essentially functioning as a no-op selection. This is only useful
    /// when the dataset will later be shuffled or transformed in place.
    ///
    /// # Arguments
    ///
    /// * `dataset` - The original dataset to select from.
    ///
    /// # Returns
    ///
    /// A new `SelectionDataset` that selects all indices from the dataset.
    pub fn new_select_all<S>(dataset: S) -> Self
    where
        S: Into<Arc<D>>,
    {
        let dataset = dataset.into();
        let size = dataset.len();
        Self::from_indices_unchecked(dataset, iota(size))
    }

    /// Creates a new selection dataset with shuffled indices.
    ///
    /// Selects every index of the dataset and shuffles them
    /// with randomness from the provided random number generator.
    ///
    /// # Arguments
    ///
    /// * `dataset` - The original dataset to select from.
    /// * `rng` - A mutable reference to a random number generator.
    ///
    /// # Returns
    ///
    /// A new `SelectionDataset` with shuffled indices.
    pub fn new_shuffled<S, R>(dataset: S, rng_source: R) -> Self
    where
        S: Into<Arc<D>>,
        R: Into<RngSource>,
    {
        let mut this = Self::new_select_all(dataset);
        this.shuffle(rng_source);
        this
    }

    /// Shuffles the indices of the dataset using a mutable random number generator.
    ///
    /// This method modifies the dataset in place, shuffling the indices.
    ///
    /// # Arguments
    ///
    /// * `rng` - A mutable reference to a random number generator.
    pub fn shuffle<R>(&mut self, rng_source: R)
    where
        R: Into<RngSource>,
    {
        let mut rng: StdRng = rng_source.into().into();
        self.indices.shuffle(&mut rng)
    }

    /// Creates a new dataset that is a slice of the current selection dataset.
    ///
    /// Slices the *selection indices* from ``[start..end]``.
    ///
    /// Independent of future shuffles on the parent, but shares the same wrapped dataset.
    ///
    ///
    /// # Arguments
    ///
    /// * `start` - The start of the range.
    /// * `end` - The end of the range (exclusive).
    // TODO: SliceArg in burn-tensor should be lifted to burn-common; this should use SliceArg.
    pub fn slice(&self, start: usize, end: usize) -> Self {
        Self::from_indices_unchecked(self.wrapped.clone(), self.indices[start..end].to_vec())
    }

    /// Split into `num` datasets by slicing the selection indices evenly.
    ///
    /// Split is done via `slice`, so the datasets share the same wrapped dataset.
    ///
    /// Independent of future shuffles on the parent, but shares the same wrapped dataset.
    ///
    /// # Arguments
    ///
    /// * `num` - The number of datasets to split into.
    ///
    /// # Returns
    ///
    /// A vector of `SelectionDataset` instances, each containing a subset of the indices.
    pub fn split(&self, num: usize) -> Vec<Self> {
        let n = self.indices.len();

        let mut current = 0;
        let mut datasets = Vec::with_capacity(num);

        let batch_size = n / num;
        for i in 0..num {
            let start = current;
            let mut end = current + batch_size;

            if i == (num - 1) {
                end = n;
            }

            let dataset = self.slice(start, end);

            current += batch_size;
            datasets.push(dataset);
        }

        datasets
    }
}

impl<D, I> Dataset<I> for SelectionDataset<D, I>
where
    D: Dataset<I>,
    I: Clone + Send + Sync,
{
    fn get(&self, index: usize) -> Option<I> {
        let index = self.indices.get(index)?;
        self.wrapped.get(*index)
    }

    fn len(&self) -> usize {
        self.indices.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FakeDataset;
    use rand::SeedableRng;

    #[test]
    fn test_iota() {
        let size = 10;
        let indices = iota(size);
        assert_eq!(indices.len(), size);
        assert_eq!(indices, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_shuffled_indices() {
        let size = 10;

        let mut rng1 = StdRng::seed_from_u64(10);
        let mut rng2 = rng1.clone();

        let mut expected = iota(size);
        expected.shuffle(&mut rng1);

        let indices = shuffled_indices(size, &mut rng2);

        assert_eq!(indices, expected);
    }

    #[should_panic(expected = "Index out of bounds for wrapped dataset size: 300 >= 27")]
    #[test]
    fn test_from_indices_checked_panics() {
        let source_dataset = FakeDataset::<String>::new(27);
        let indices: Vec<usize> = vec![15, 1, 12, 300];
        SelectionDataset::from_indices_checked(source_dataset, indices);
    }

    #[test]
    fn test_checked_selection_dataset() {
        let source_dataset = FakeDataset::<String>::new(27);

        let indices: Vec<usize> = vec![15, 1, 12, 12];
        let expected: Vec<String> = indices
            .iter()
            .map(|i| source_dataset.get(*i).unwrap())
            .collect();

        let selection = SelectionDataset::from_indices_checked(source_dataset, indices.clone());

        assert_eq!(&selection.indices, &indices);

        let items = selection.iter().collect::<Vec<_>>();

        assert_eq!(items, expected);
    }

    #[test]
    fn test_shuffled_dataset() {
        let dataset = FakeDataset::<String>::new(27);
        let source_items = dataset.iter().collect::<Vec<_>>();

        let selection = SelectionDataset::new_shuffled(dataset, 42);

        let indices = shuffled_indices(source_items.len(), &mut StdRng::seed_from_u64(42));

        assert_eq!(&selection.indices, &indices);
        assert_eq!(selection.len(), source_items.len());

        let expected_items: Vec<_> = indices
            .iter()
            .map(|&i| source_items[i].to_string())
            .collect();
        assert_eq!(&selection.iter().collect::<Vec<_>>(), &expected_items);
    }

    #[test]
    fn test_slice() {
        let dataset = FakeDataset::<String>::new(27);
        let source_items = dataset.iter().collect::<Vec<_>>();

        let selection = SelectionDataset::new_select_all(dataset);

        let start = 5;
        let end = 15;
        let sliced_selection = selection.slice(start, end);

        assert_eq!(sliced_selection.len(), end - start);

        #[allow(clippy::needless_range_loop)]
        for i in start..end {
            assert_eq!(
                sliced_selection.get(i - start),
                Some(source_items[i].to_string())
            );
        }
    }

    #[test]
    fn test_split() {
        let dataset = FakeDataset::<String>::new(28);
        let source_items = dataset.iter().collect::<Vec<_>>();

        let selection = SelectionDataset::new_select_all(dataset);

        let split_contents: Vec<Vec<_>> = selection
            .split(3)
            .iter()
            .map(|d| d.iter().collect::<Vec<_>>())
            .collect();
        assert_eq!(
            split_contents,
            vec![
                source_items[0..9].to_vec(),
                source_items[9..18].to_vec(),
                source_items[18..28].to_vec(),
            ]
        );
    }
}
