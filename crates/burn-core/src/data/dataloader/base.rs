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

    /// The current resource id assigned by the strategy.
    pub resource_id: Option<usize>,
}

/// A data loader iterator that can be used to iterate over a data loader.
pub trait DataLoaderIterator<O>: Iterator<Item = O> {
    /// The strategy used to assign data across resources/devices.
    type Strategy: AssignmentStrategy;
    /// Returns the progress of the data loader.
    fn progress(&self) -> Progress;
    /// Returns the state of the data loader.
    fn state(&self) -> State;
}

/// A data loader that can be used to iterate over a dataset.
pub trait DataLoader<O>: Send {
    /// Returns a boxed [iterator](DataLoaderIterator) to iterate over the data loader.
    // For now, the data loader has a fixed round-robin assignment strategy.
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<O, Strategy = RoundRobinAssignment> + 'a>;
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

/// A strategy for assigning items, potentially distributing them across devices or other resources.
pub trait AssignmentStrategy: Send {
    /// Selects the next resource to assign the item to.
    fn step(&mut self);

    /// Returns the current resource id.
    fn current(&self) -> usize;
}

/// A strategy for assigning items to resources in a round-robin fashion.
/// It cycles through the available resources, ensuring that batches are
/// assigned in turn to each resource, balancing the load evenly across them.
#[derive(Clone, Debug)]
pub struct RoundRobinAssignment {
    total: usize,
    current: usize,
}

impl RoundRobinAssignment {
    /// Create a new assignment strategy for the specified number of resources.
    pub fn new(num_resources: usize) -> Self {
        RoundRobinAssignment {
            total: num_resources,
            current: 0,
        }
    }
}

impl AssignmentStrategy for RoundRobinAssignment {
    fn step(&mut self) {
        // Round-robin selection (alternate for each item)
        self.current += 1;
    }

    fn current(&self) -> usize {
        self.current % self.total
    }
}
