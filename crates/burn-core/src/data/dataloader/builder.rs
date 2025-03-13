use super::{
    batcher::DynBatcher, BatchStrategy, DistributionStrategy, FixBatchStrategy,
    LazyBatchDataLoader, LazyDataLoader,
};
use burn_dataset::Dataset;
use burn_tensor::backend::Backend;
use rand::{rngs::StdRng, SeedableRng};
use std::sync::Arc;

/// A builder for data loaders.
pub struct DataLoaderBuilder<B: Backend, I, O> {
    strategy: Option<Box<dyn BatchStrategy<I>>>,
    batcher: Box<dyn DynBatcher<B, I, O>>,
    num_threads: Option<usize>,
    shuffle: Option<u64>,
    distributor: Option<Box<dyn DistributionStrategy<Resource = B::Device>>>,
}

impl<B, I, O> DataLoaderBuilder<B, I, O>
where
    B: Backend,
    I: Send + Sync + Clone + std::fmt::Debug + 'static,
    O: Send + Clone + std::fmt::Debug + 'static,
{
    /// Creates a new data loader builder.
    ///
    /// # Arguments
    ///
    /// * `batcher` - The batcher.
    ///
    /// # Returns
    ///
    /// The data loader builder.
    pub fn new<Bt>(batcher: Bt) -> Self
    where
        Bt: DynBatcher<B, I, O> + 'static,
    {
        Self {
            batcher: Box::new(batcher),
            strategy: None,
            num_threads: None,
            shuffle: None,
            distributor: None,
        }
    }

    /// Sets the batch size to a fix number.The [fix batch strategy](FixBatchStrategy)
    /// will be used.
    ///
    /// # Arguments
    ///
    /// * `batch_size` - The batch size.
    ///
    /// # Returns
    ///
    /// The data loader builder.
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.strategy = Some(Box::new(FixBatchStrategy::new(batch_size)));
        self
    }

    /// Sets the seed for shuffling.
    ///
    /// Each time the dataloader starts a new iteration, the dataset will be shuffled.
    ///
    /// # Arguments
    ///
    /// * `seed` - The seed.
    ///
    /// # Returns
    ///
    /// The data loader builder.
    pub fn shuffle(mut self, seed: u64) -> Self {
        self.shuffle = Some(seed);
        self
    }

    /// Sets the number of workers.
    ///
    /// # Arguments
    ///
    /// * `num_workers` - The number of workers.
    ///
    /// # Returns
    ///
    /// The data loader builder.
    pub fn num_workers(mut self, num_workers: usize) -> Self {
        self.num_threads = Some(num_workers);
        self
    }

    /// Sets the data loader device distribution/selection strategy for a batch.
    ///
    /// # Arguments
    ///
    /// * `distributor` - The device distribution strategy.
    ///
    /// # Returns
    ///
    /// The data loader builder.
    pub fn distributor<D>(mut self, distributor: D) -> Self
    where
        D: DistributionStrategy<Resource = B::Device>,
    {
        self.distributor = Some(distributor.clone_dyn());
        self
    }

    /// Builds the data loader.
    ///
    /// # Arguments
    ///
    /// * `dataset` - The dataset.
    ///
    /// # Returns
    ///
    /// The data loader.
    pub fn build<D>(self, dataset: D) -> Arc<dyn LazyDataLoader<O, Resource = B::Device>>
    where
        D: Dataset<I> + 'static,
    {
        let dataset = Arc::new(dataset);

        let rng = self.shuffle.map(StdRng::seed_from_u64);
        let strategy = match self.strategy {
            Some(strategy) => strategy,
            None => Box::new(FixBatchStrategy::new(1)),
        };

        Arc::new(LazyBatchDataLoader::new(
            strategy,
            dataset,
            self.batcher,
            self.distributor,
            rng,
            self.num_threads.unwrap_or(0),
        ))
    }
}
