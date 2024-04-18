use std::{cmp::max, num::NonZeroUsize};

use crate::Dataset;

impl<'a, I> WindowDataset<'a, I> {
    /// Creates a new `WindowDataset` instance. The windows overlap.
    /// Is empty if the input `Dataset` is shorter than `size`.
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
    pub size: NonZeroUsize,
    dataset: &'a dyn Dataset<I>,
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
    ///
    /// # Returns
    ///
    /// A size representing the number of windows.
    fn len(&self) -> usize {
        let len = self.dataset.len() as isize - self.size.get() as isize + 1;
        max(len, 0) as usize
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::{Dataset, InMemDataset};

    #[rstest]
    pub fn windows_to_vec_should_be_equal() {
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
    #[should_panic]
    pub fn windows_should_panic() {
        let items = [1, 2].to_vec();
        let dataset = InMemDataset::new(items.clone());

        dataset.windows(0);
    }

    #[rstest]
    pub fn len_should_be_equal() {
        let dataset = InMemDataset::new([1, 2, 3, 4].to_vec());

        let result = dataset.windows(2).len();

        assert_eq!(result, 3);
    }

    #[rstest]
    pub fn len_should_be_zero() {
        let dataset = InMemDataset::new([1, 2].to_vec());

        let result = dataset.windows(4).len();

        assert_eq!(result, 0);
    }

    #[rstest]
    pub fn get_should_be_equal() {
        let dataset = InMemDataset::new([1, 2, 3, 4].to_vec());
        let expected = Some([1, 2, 3].to_vec());

        let result = dataset.windows(3).get(0);

        assert_eq!(result, expected);
    }

    #[rstest]
    pub fn get_should_be_none() {
        let dataset = InMemDataset::new([1, 2].to_vec());

        let result = dataset.windows(4).get(0);

        assert_eq!(result, None);
    }
}
