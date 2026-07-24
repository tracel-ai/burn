use crate::dataset::{Dataset, DatasetError};
use std::error::Error;
use std::iter::Iterator;

/// Dataset iterator.
pub struct DatasetIterator<'a, I, E = DatasetError>
where
    E: Error + Send + Sync + 'static,
{
    current: usize,
    dataset: &'a dyn Dataset<I, E>,
    len: usize,
}

impl<'a, I, E> DatasetIterator<'a, I, E>
where
    E: Error + Send + Sync + 'static,
{
    /// Creates a new dataset iterator.
    pub fn new<D>(dataset: &'a D) -> Self
    where
        D: Dataset<I, E>,
    {
        DatasetIterator {
            current: 0,
            dataset,
            len: dataset.len(),
        }
    }
}

impl<I, E> Iterator for DatasetIterator<'_, I, E>
where
    E: Error + Send + Sync + 'static,
{
    type Item = Result<I, E>;

    fn next(&mut self) -> Option<Result<I, E>> {
        if self.current >= self.len {
            return None;
        }

        let index = self.current;
        self.current += 1;

        Some(self.dataset.get(index))
    }
}
