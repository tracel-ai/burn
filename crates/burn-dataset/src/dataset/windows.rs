use std::usize;

use crate::Dataset;

impl<'a, I> DatasetWindows<'a, I> {
    pub fn new<D>(dataset: &'a D, window_size: usize) -> Self
    where
        D: Dataset<I>,
    {
        DatasetWindows {
            window_size,
            dataset,
        }
    }
}

pub struct DatasetWindows<'a, I> {
    dataset: &'a dyn Dataset<I>,
    window_size: usize,
}

impl<'a, I> Dataset<Vec<I>> for DatasetWindows<'a, I> {
    fn get(&self, index: usize) -> Option<Vec<I>> {
        (index..index + self.window_size)
            .map(|x| self.dataset.get(x))
            .collect()
    }

    fn len(&self) -> usize {
        self.dataset.len() - self.window_size + 1
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::{Dataset, InMemDataset};

    #[rstest]
    pub fn get_windows() {
        let dataset = InMemDataset::new([1, 2, 3, 4, 5].to_vec());
        let windows_dataset = dataset.windows(3);
        let expected = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
            .map(|x| x.to_vec())
            .to_vec();

        assert_eq!(windows_dataset.len(), 3);
        assert_eq!(windows_dataset.iter().collect::<Vec<Vec<i32>>>(), expected);
    }
}
