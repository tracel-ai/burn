use std::error::Error;
use std::sync::Arc;

use crate::{DatasetError, DatasetIterator};

/// The dataset trait defines a basic collection of items with a predefined size.
///
/// # Panics
///
/// `get` panics if `index >= len()`, matching slice/`Vec` indexing conventions. `Err(_)` is
/// reserved for genuine retrieval failures on an in-bounds index (e.g. an I/O or deserialization
/// failure).
pub trait Dataset<I, E = DatasetError>: Send + Sync
where
    E: Error + Send + Sync + 'static,
{
    /// Gets the item at the given index.
    ///
    /// # Panics
    ///
    /// Panics if `index >= len()`.
    fn get(&self, index: usize) -> Result<I, E>;

    /// Gets the number of items in the dataset.
    fn len(&self) -> usize;

    /// Checks if the dataset is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns an iterator over the dataset.
    fn iter(&self) -> DatasetIterator<'_, I, E>
    where
        Self: Sized,
    {
        DatasetIterator::new(self)
    }
}

impl<D, I, E> Dataset<I, E> for Arc<D>
where
    D: Dataset<I, E>,
    E: Error + Send + Sync + 'static,
{
    fn get(&self, index: usize) -> Result<I, E> {
        self.as_ref().get(index)
    }

    fn len(&self) -> usize {
        self.as_ref().len()
    }
}

impl<I, E> Dataset<I, E> for Arc<dyn Dataset<I, E>>
where
    E: Error + Send + Sync + 'static,
{
    fn get(&self, index: usize) -> Result<I, E> {
        self.as_ref().get(index)
    }

    fn len(&self) -> usize {
        self.as_ref().len()
    }
}

impl<D, I, E> Dataset<I, E> for Box<D>
where
    D: Dataset<I, E>,
    E: Error + Send + Sync + 'static,
{
    fn get(&self, index: usize) -> Result<I, E> {
        self.as_ref().get(index)
    }

    fn len(&self) -> usize {
        self.as_ref().len()
    }
}

impl<I, E> Dataset<I, E> for Box<dyn Dataset<I, E>>
where
    E: Error + Send + Sync + 'static,
{
    fn get(&self, index: usize) -> Result<I, E> {
        self.as_ref().get(index)
    }

    fn len(&self) -> usize {
        self.as_ref().len()
    }
}
