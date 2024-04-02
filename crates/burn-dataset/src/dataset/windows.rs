use std::{cmp::min, usize};

use crate::Dataset;

impl<'a, I> WindowsDataset<'a, I> {
    pub fn new<D>(dataset: &'a D, window_size: usize) -> Self
    where
        D: Dataset<I>,
    {
        WindowsDataset {
            window_size,
            dataset,
        }
    }
}

pub struct WindowsDataset<'a, I> {
    dataset: &'a dyn Dataset<I>,
    window_size: usize,
}

impl<'a, I> Dataset<Vec<I>> for WindowsDataset<'a, I> {
    fn get(&self, index: usize) -> Option<Vec<I>> {
        (index..(index + self.window_size))
            .map(|x| self.dataset.get(x))
            .collect()
    }

    fn len(&self) -> usize {
        self.dataset.len() - self.window_size
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::{Dataset, InMemDataset};

    #[rstest]
    pub fn windows() {
        let dataset = InMemDataset::new([1, 2, 3, 4, 5].to_vec());
        let windows_dataset = dataset.windows(2);
        let expected = [[1, 2].to_vec(), [2, 3].to_vec(), [3,4].to_vec(),[4,5].to_vec()].to_vec();

        assert_eq!(windows_dataset.iter().collect::<Vec<Vec<i32>>>(),expected);
    }
}
