use core::cell::OnceCell;
use std::sync::Arc;

use burn_dataset::{transform::PartialDataset, Dataset};
use burn_tensor::backend::Backend;
use rand::{
    distr::{Distribution, StandardUniform},
    rngs::StdRng,
    SeedableRng,
};

use super::{
    batcher::DynBatcher, BatchDataLoader, BatchDispatcher, BatchStrategy, DataLoader,
    DataLoaderIterator,
};

/// A lazy data loader that can be split over the total number of items.
pub trait LazyDataLoader<O>: DataLoader<O> {
    /// The resource type for the dispatching strategy split.
    type Resource: Send + Clone;
    /// Splits a data loader into multiple partial data loaders.
    ///
    /// # Arguments
    ///
    /// * `dispatchers` - The dispatching strategy for each data loader.
    ///
    /// # Returns
    /// A vector of lazy data loaders, each initialized with one of the provided dispatching strategies.
    /// Each returned data loader represents a portion of the original dataset and can be independently
    /// processed. The number of returned data loaders matches the number of dispatching strategies provided.
    fn split(
        &self,
        dispatchers: Vec<Box<dyn BatchDispatcher<Resource = Self::Resource>>>,
    ) -> Vec<Box<dyn LazyDataLoader<O, Resource = Self::Resource>>>;
}

/// A wrapper struct that lazily initializes either a [`BatchDataLoader`] or
/// [`MultiThreadDataLoader`](super::MultiThreadDataLoader).
pub struct LazyBatchDataLoader<B: Backend, I, O> {
    // Configuration parameters needed for initialization
    strategy: Box<dyn BatchStrategy<I>>,
    dataset: Arc<dyn Dataset<I>>,
    batcher: Box<dyn DynBatcher<B, I, O>>,
    dispatcher: Arc<dyn BatchDispatcher<Resource = B::Device>>,
    rng: Option<rand::rngs::StdRng>,
    num_threads: usize,

    // The lazily initialized loader
    inner: OnceCell<Box<dyn DataLoader<O>>>,
}

impl<B: Backend, I, O> Clone for LazyBatchDataLoader<B, I, O> {
    fn clone(&self) -> Self {
        Self {
            strategy: self.strategy.clone_dyn(),
            dataset: self.dataset.clone(),
            batcher: self.batcher.clone_dyn(),
            dispatcher: self.dispatcher.clone(),
            rng: self.rng.clone(),
            num_threads: self.num_threads,
            inner: OnceCell::new(),
        }
    }
}

impl<B: Backend, I, O> LazyBatchDataLoader<B, I, O>
where
    I: Send + Sync + Clone + 'static,
    O: Send + Clone + std::fmt::Debug + 'static,
{
    /// Creates a new lazy batch data loader.
    ///
    /// # Arguments
    ///
    /// * `strategy` - The batch strategy.
    /// * `dataset` - The dataset.
    /// * `batcher` - The batcher.
    /// * `dispatcher` - The resource dispatching strategy.
    /// * `rng`     - The rng determining if the dataset is shuffled each time a dataloader
    ///               iterator is created.
    /// * `num_threads` - The number of threads.
    ///
    /// # Returns
    ///
    /// The batch data loader.
    pub fn new(
        strategy: Box<dyn BatchStrategy<I>>,
        dataset: Arc<dyn Dataset<I>>,
        batcher: Box<dyn DynBatcher<B, I, O>>,
        dispatcher: Arc<dyn BatchDispatcher<Resource = B::Device>>,
        rng: Option<rand::rngs::StdRng>,
        num_threads: usize,
    ) -> Self {
        Self {
            strategy,
            dataset,
            batcher,
            dispatcher,
            rng,
            num_threads,
            inner: OnceCell::new(),
        }
    }

    /// Force initialization if needed.
    fn initialize(&self) -> &dyn DataLoader<O> {
        self.inner
            .get_or_init(|| {
                if self.num_threads > 1 {
                    Box::new(BatchDataLoader::multi_thread(
                        self.strategy.clone_dyn(),
                        self.dataset.clone(),
                        self.batcher.clone_dyn(),
                        self.num_threads,
                        self.dispatcher.clone(),
                        self.rng.clone(),
                    ))
                } else {
                    // Create a regular BatchDataLoader
                    Box::new(BatchDataLoader::new(
                        self.strategy.clone_dyn(),
                        self.dataset.clone(),
                        self.batcher.clone_dyn(),
                        self.dispatcher.clone(),
                        self.rng.clone(),
                    ))
                }
            })
            .as_ref()
    }
}

impl<B: Backend, I, O> DataLoader<O> for LazyBatchDataLoader<B, I, O>
where
    I: Send + Sync + Clone + 'static,
    O: Send + Clone + std::fmt::Debug + 'static,
{
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<O> + 'a> {
        // This will initialize the loader if it hasn't been initialized yet
        let loader = self.initialize();
        loader.iter()
    }

    fn num_items(&self) -> usize {
        // For num_items, we can directly use the dataset size without
        // necessarily initializing the full loader
        self.dataset.len()
    }
}

// By having a lazy dataloader we can easily split the data across multiple threads, devices, etc. down the line.
impl<B: Backend, I, O> LazyDataLoader<O> for LazyBatchDataLoader<B, I, O>
where
    I: Send + Sync + Clone + 'static,
    O: Send + Clone + std::fmt::Debug + 'static,
{
    type Resource = B::Device;

    fn split(
        &self,
        dispatchers: Vec<Box<dyn BatchDispatcher<Resource = Self::Resource>>>,
    ) -> Vec<Box<dyn LazyDataLoader<O, Resource = Self::Resource>>> {
        let num_splits = dispatchers.len();
        let datasets = PartialDataset::split(self.dataset.clone(), num_splits);

        let mut dataloaders = Vec::with_capacity(num_splits);

        // Create more rngs from the first one, one for each new dataloader.
        let mut rng = self.rng.clone();
        let rngs = (0..num_splits).map(|_| {
            rng.as_mut()
                .map(|rng| StdRng::seed_from_u64(Distribution::sample(&StandardUniform, rng)))
        });

        for ((dataset, rng), dispatcher) in datasets.into_iter().zip(rngs).zip(dispatchers) {
            let dataloader = LazyBatchDataLoader::new(
                self.strategy.clone_dyn(),
                Arc::new(dataset),
                self.batcher.clone_dyn(),
                Arc::from(dispatcher),
                rng,
                self.num_threads,
            );
            let dataloader: Box<dyn LazyDataLoader<_, Resource = Self::Resource>> =
                Box::new(dataloader);
            dataloaders.push(dataloader);
        }
        dataloaders
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::data::dataloader::batcher::TestBatcher;
    use crate::data::dataloader::FixBatchStrategy;
    use crate::data::dataloader::FixedDispatcher;
    use crate::data::dataset::FakeDataset;

    #[test]
    fn test_batch_dataloader_lazy() {
        let batcher = Box::new(TestBatcher::new());
        let dispatcher = Arc::new(FixedDispatcher::new(vec![Default::default()]));
        let dataset = Arc::new(FakeDataset::<String>::new(27));
        let dataloader = LazyBatchDataLoader::new(
            Box::new(FixBatchStrategy::new(5)),
            dataset.clone(),
            batcher,
            dispatcher,
            None,
            0,
        );

        let mut items_dataset = HashSet::new();
        let mut items_dataloader = HashSet::new();

        for item in dataset.iter() {
            items_dataset.insert(item);
        }

        for items in dataloader.iter() {
            for item in items {
                items_dataloader.insert(item);
            }
        }

        assert_eq!(items_dataset, items_dataloader);
    }

    #[test]
    fn test_multi_thread_batch_dataloader_lazy() {
        let batcher = Box::new(TestBatcher::new());
        let dispatcher = Arc::new(FixedDispatcher::new(vec![Default::default()]));
        let dataset = Arc::new(FakeDataset::<String>::new(27));
        let dataloader_single_thread = LazyBatchDataLoader::new(
            Box::new(FixBatchStrategy::new(5)),
            dataset.clone(),
            batcher.clone_dyn(),
            dispatcher.clone(),
            None,
            0,
        );
        let dataloader_multi_thread = LazyBatchDataLoader::new(
            Box::new(FixBatchStrategy::new(5)),
            dataset,
            batcher,
            dispatcher,
            None,
            4,
        );

        let mut items_single_thread = HashSet::new();
        let mut items_multi_thread = HashSet::new();

        for items in dataloader_single_thread.iter() {
            for item in items {
                items_single_thread.insert(item);
            }
        }

        for items in dataloader_multi_thread.iter() {
            for item in items {
                items_multi_thread.insert(item);
            }
        }

        assert_eq!(items_single_thread, items_multi_thread);
    }

    #[test]
    fn test_multi_thread_batch_dataloader_split() {
        let batcher = Box::new(TestBatcher::new());
        let dispatcher = Arc::new(FixedDispatcher::new(vec![Default::default()]));
        let dataset = Arc::new(FakeDataset::<String>::new(27));
        let dataloader_single_thread = LazyBatchDataLoader::new(
            Box::new(FixBatchStrategy::new(5)),
            dataset.clone(),
            batcher.clone_dyn(),
            dispatcher.clone(),
            None,
            0,
        );

        let dispatcher_multi = FixedDispatcher::new(vec![Default::default(), Default::default()]);
        let dataloader_split = LazyBatchDataLoader::new(
            Box::new(FixBatchStrategy::new(5)),
            dataset,
            batcher,
            dispatcher,
            None,
            4,
        )
        .split(vec![
            dispatcher_multi.clone_dyn(),
            dispatcher_multi.with_fixed(1).clone_dyn(),
        ]);

        let mut items_single_thread = HashSet::new();
        let mut items_multi_thread_split = HashSet::new();

        for items in dataloader_single_thread.iter() {
            for item in items {
                items_single_thread.insert(item);
            }
        }

        for dataloader_multi_thread in dataloader_split {
            for items in dataloader_multi_thread.iter() {
                for item in items {
                    items_multi_thread_split.insert(item);
                }
            }
        }

        assert_eq!(items_single_thread, items_multi_thread_split);
    }

    #[test]
    fn test_batch_dataloader_lazy_split_after_iter() {
        let batcher = Box::new(TestBatcher::new());
        let dispatcher = Arc::new(FixedDispatcher::new(vec![Default::default()]));
        let dataset = Arc::new(FakeDataset::<String>::new(27));
        let dataloader = LazyBatchDataLoader::new(
            Box::new(FixBatchStrategy::new(5)),
            dataset.clone(),
            batcher,
            dispatcher,
            None,
            0,
        );

        let mut items_single_thread = HashSet::new();
        let mut items_single_thread_split = HashSet::new();

        for items in dataloader.iter() {
            for item in items {
                items_single_thread.insert(item);
            }
        }

        let dispatcher_multi = FixedDispatcher::new(vec![Default::default(), Default::default()]);
        let dataloader_split = dataloader.split(vec![
            dispatcher_multi.clone_dyn(),
            dispatcher_multi.with_fixed(1).clone_dyn(),
        ]);

        for dataloader in dataloader_split {
            for items in dataloader.iter() {
                for item in items {
                    items_single_thread_split.insert(item);
                }
            }
        }

        assert_eq!(items_single_thread, items_single_thread_split);
    }
}
