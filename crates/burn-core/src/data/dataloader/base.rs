pub use crate::data::dataset::{Dataset, DatasetIterator};
use core::iter::Iterator;

/// A progress struct that can be used to track the progress of a data loader.
#[derive(new, Clone, Debug)]
pub struct Progress {
    /// The number of items that have been processed.
    pub items_processed: usize,

    /// The total number of items that need to be processed.
    pub items_total: usize,
}

/// Represents the current state of the data loader, including progress tracking
/// and the last assigned resource for batch distribution.
#[derive(new, Clone, Debug)]
pub struct State {
    /// The data loader progress.
    pub progress: Progress,

    /// The last resource id assigned by the strategy.
    pub resource_id: Option<usize>,
}

/// A data loader iterator that can be used to iterate over a data loader.
pub trait DataLoaderIterator<O>: Iterator<Item = O> {
    /// Returns the progress of the data loader.
    fn progress(&self) -> Progress;
    /// Returns the state of the data loader.
    fn state(&self) -> State;
}

/// A data loader that can be used to iterate over a dataset.
pub trait DataLoader<O>: Send {
    /// Returns a boxed [iterator](DataLoaderIterator) to iterate over the data loader.
    // For now, the data loader has a fixed round-robin assignment strategy.
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<O> + 'a>;
    /// The number of items (not the number of batches nor the number of iterations),
    /// corresponding to the items_total of the progress returned by the iterator.
    fn num_items(&self) -> usize;
}

/// A super trait for [dataloader](DataLoader) that allows it to be cloned dynamically.
///
/// Any dataloader that implements [Clone] should also implement this automatically.
pub trait DynDataLoader<O>: DataLoader<O> {
    /// Clone the dataloader and returns a new one.
    fn clone_dyn(&self) -> Box<dyn DynDataLoader<O>>;
}

impl<D, O> DynDataLoader<O> for D
where
    D: DataLoader<O> + Clone + 'static,
{
    fn clone_dyn(&self) -> Box<dyn DynDataLoader<O>> {
        Box::new(self.clone())
    }
}
