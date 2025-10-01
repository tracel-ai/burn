use burn_tensor::backend::Backend;

pub use crate::data::dataset::{Dataset, DatasetIterator};
use core::iter::Iterator;
use std::sync::Arc;

/// A progress struct that can be used to track the progress of a data loader.
#[derive(new, Clone, Debug)]
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
pub trait DataLoader<B: Backend, O>: Send + Sync {
    /// Returns a boxed [iterator](DataLoaderIterator) to iterate over the data loader.
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<O> + 'a>;

    /// The number of items (not the number of batches nor the number of iterations),
    /// corresponding to the items_total of the progress returned by the iterator.
    fn num_items(&self) -> usize;

    /// Move the data loader to the given device, ensuring the batches are assigned to the correct device.
    fn to_device(&self, device: &B::Device) -> Arc<dyn DataLoader<B, O>>;

    /// Returns a new data loader containing a subset of the data.
    ///
    /// The subset includes items from `start` (inclusive) to `end` (exclusive),
    /// preserving the batch size and ordering of the original data loader.
    ///
    /// # Arguments
    ///
    /// * `start` - The starting index of the subset (inclusive).
    /// * `end` - The ending index of the subset (exclusive).
    ///
    /// # Returns
    ///
    /// A boxed [`DataLoader`] instance containing only the specified range.
    fn slice(&self, start: usize, end: usize) -> Arc<dyn DataLoader<B, O>>;
}
