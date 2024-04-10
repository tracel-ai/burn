use std::num::NonZeroUsize;
use std::usize;

use crate::Dataset;

impl<'a, I> WindowDataset<'a, I> {
    /// Creates a new `WindowDataset` instance.
    ///
    /// # Parameters
    ///
    /// - `dataset`: The dataset over which windows will be created.
    /// - `size`: The size of the window.
    ///
    /// # Returns
    ///
    /// A `WindowDataset` instance.
    pub fn new<D>(dataset: &'a D, size: NonZeroUsize) -> Self
    where
        D: Dataset<I>,
    {
        WindowDataset { size, dataset }
    }
}

/// Dataset designed to work with overlapping windows of data.
pub struct WindowDataset<'a, I> {
    dataset: &'a dyn Dataset<I>,
    size: NonZeroUsize,
}

impl<'a, I> Dataset<Vec<I>> for WindowDataset<'a, I> {
    /// Retrieves a window of items from the dataset.
    ///
    /// # Parameters
    ///
    /// - `index`: The index of the window.
    ///
    /// # Returns
    ///
    /// A vector containing the items of the window.
    fn get(&self, index: usize) -> Option<Vec<I>> {
        (index..index + self.size.get())
            .map(|x| self.dataset.get(x))
            .collect()
    }

    /// Retrieves the number of windows in the dataset.
    fn len(&self) -> usize {
        self.dataset.len() - self.size.get() + 1
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::{Dataset, InMemDataset};

    #[rstest]
    pub fn windows_should_match() {
        let items = [1, 2, 3, 4, 5].to_vec();
        let dataset = InMemDataset::new(items.clone());
        let expected = items
            .windows(3)
            .map(|x| x.to_vec())
            .collect::<Vec<Vec<i32>>>();

        let result = dataset.windows(3).iter().collect::<Vec<Vec<i32>>>();

        assert_eq!(result, expected);
    }

    #[rstest]
    pub fn len_should_match() {
        let dataset = InMemDataset::new([1, 2, 3].to_vec());

        let result = dataset.windows(2).len();

        assert_eq!(result, 2);
    }
}
