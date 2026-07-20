use crate::dataset::Dataset;
use std::iter::Iterator;

/// Dataset iterator.
pub struct DatasetIterator<'a, I> {
    current: usize,
    dataset: &'a dyn Dataset<I>,
    len: usize,
}

impl<'a, I> DatasetIterator<'a, I> {
    /// Creates a new dataset iterator.
    pub fn new<D>(dataset: &'a D) -> Self
    where
        D: Dataset<I>,
    {
        DatasetIterator {
            current: 0,
            dataset,
            len: dataset.len(),
        }
    }
}

impl<I> Iterator for DatasetIterator<'_, I> {
    type Item = I;

    fn next(&mut self) -> Option<I> {
        while self.current < self.len {
            let index = self.current;
            self.current += 1;

            if let Some(item) = self.dataset.get(index) {
                return Some(item);
            }
        }
        None
    }
}
