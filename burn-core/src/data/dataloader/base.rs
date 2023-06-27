pub use crate::data::dataset::{Dataset, DatasetIterator};
use core::iter::Iterator;

/// A progress struct that can be used to track the progress of a data loader.
#[derive(Clone, Debug)]
pub struct Progress {
    /// The number of items that have been processed.
    pub items_processed: usize,

    /// The total number of items that need to be processed.
    pub items_total: usize,
}

/// A data loader iterator that can be used to iterate over a data loader.
pub trait DataLoaderIterator<O>: Iterator<Item = O> {
    /// Returns the progress of the data loader.
    fn progress(&self) -> Progress;
}

/// A data loader that can be used to iterate over a dataset.
pub trait DataLoader<O> {
    /// Returns a boxed [iterator](DataLoaderIterator) to iterate over the data loader.
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<O> + 'a>;
}
