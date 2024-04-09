use std::num::NonZeroUsize;
use std::usize;

use crate::Dataset;

impl<'a, I> DatasetWindows<'a, I> {
    pub fn new<D>(dataset: &'a D, size: NonZeroUsize) -> Self
    where
        D: Dataset<I>,
    {
        DatasetWindows { size, dataset }
    }
}

pub struct DatasetWindows<'a, I> {
    dataset: &'a dyn Dataset<I>,
    size: NonZeroUsize,
}

impl<'a, I> Dataset<Vec<I>> for DatasetWindows<'a, I> {
    fn get(&self, index: usize) -> Option<Vec<I>> {
        (index..index + self.size.get())
            .map(|x| self.dataset.get(x))
            .collect()
    }

    fn len(&self) -> usize {
        self.dataset.len() - self.size.get() + 1
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::{Dataset, InMemDataset};

    #[rstest]
    pub fn get_windows() {
        let dataset = InMemDataset::new([1, 2, 3, 4, 5].to_vec());
        let expected = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
            .map(|x| x.to_vec())
            .to_vec();

        let windows = dataset.windows(3);

        assert_eq!(windows.len(), 3);
        assert_eq!(windows.iter().collect::<Vec<Vec<i32>>>(), expected);
    }
}
