use std::{num::NonZeroUsize, sync::Arc};

use crate::DatasetIterator;

use super::windows::DatasetWindows;

/// The dataset trait defines a basic collection of items with a predefined size.
pub trait Dataset<I>: Send + Sync {
    /// Gets the item at the given index.
    fn get(&self, index: usize) -> Option<I>;

    /// Gets the number of items in the dataset.
    fn len(&self) -> usize;

    /// Checks if the dataset is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns an iterator over the dataset.
    fn iter(&self) -> DatasetIterator<'_, I>
    where
        Self: Sized,
    {
        DatasetIterator::new(self)
    }

    /// Returns a new `Dataset` of all the windows of length `size`. The windows overlap.
    ///
    /// # Panics
    ///
    /// Panics if `size` is 0.    
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::burn_dataset::{Dataset,InMemDataset};
    /// let dataset = InMemDataset::new([1, 2, 3].to_vec());
    ///
    /// let windows = dataset.windows(2);

    /// assert_eq!(windows.len(), 2);
    /// assert_eq!(
    ///     windows.iter().collect::<Vec<Vec<i32>>>(),
    ///      [[1, 2], [2, 3]]
    ///          .map(|x| x.to_vec())
    ///          .to_vec()
    ///  );
    /// ```
    fn windows(&self, size: usize) -> DatasetWindows<'_, I>
    where
        Self: Sized,
    {
        let size = NonZeroUsize::new(size).expect("window size must be non-zero");
        DatasetWindows::new(self, size)
    }
}

impl<D, I> Dataset<I> for Arc<D>
where
    D: Dataset<I>,
{
    fn get(&self, index: usize) -> Option<I> {
        self.as_ref().get(index)
    }

    fn len(&self) -> usize {
        self.as_ref().len()
    }
}

impl<I> Dataset<I> for Arc<dyn Dataset<I>> {
    fn get(&self, index: usize) -> Option<I> {
        self.as_ref().get(index)
    }

    fn len(&self) -> usize {
        self.as_ref().len()
    }
}

impl<D, I> Dataset<I> for Box<D>
where
    D: Dataset<I>,
{
    fn get(&self, index: usize) -> Option<I> {
        self.as_ref().get(index)
    }

    fn len(&self) -> usize {
        self.as_ref().len()
    }
}

impl<I> Dataset<I> for Box<dyn Dataset<I>> {
    fn get(&self, index: usize) -> Option<I> {
        self.as_ref().get(index)
    }

    fn len(&self) -> usize {
        self.as_ref().len()
    }
}
