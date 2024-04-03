use super::{batcher::DynBatcher, BatchDataLoader, BatchStrategy, DataLoader, FixBatchStrategy};
use burn_dataset::Dataset;
use rand::{rngs::StdRng, SeedableRng};
use std::sync::Arc;

/// A builder for data loaders.
pub struct DataLoaderBuilder<I, O> {
    strategy: Option<Box<dyn BatchStrategy<I>>>,
    batcher: Box<dyn DynBatcher<I, O>>,
    num_threads: Option<usize>,
    shuffle: Option<u64>,
}

impl<I, O> DataLoaderBuilder<I, O>
where
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
    pub fn new<B>(batcher: B) -> Self
    where
        B: DynBatcher<I, O> + 'static,
    {
        Self {
            batcher: Box::new(batcher),
            strategy: None,
            num_threads: None,
            shuffle: None,
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

    /// Builds the data loader.
    ///
    /// # Arguments
    ///
    /// * `dataset` - The dataset.
    ///
    /// # Returns
    ///
    /// The data loader.
    pub fn build<D>(self, dataset: D) -> Arc<dyn DataLoader<O>>
    where
        D: Dataset<I> + 'static,
    {
        let dataset = Arc::new(dataset);

        let rng = self.shuffle.map(StdRng::seed_from_u64);
        let strategy = match self.strategy {
            Some(strategy) => strategy,
            None => Box::new(FixBatchStrategy::new(1)),
        };
        if let Some(num_threads) = self.num_threads {
            return Arc::new(BatchDataLoader::multi_thread(
                strategy,
                dataset,
                self.batcher,
                num_threads,
                rng,
            ));
        }

        Arc::new(BatchDataLoader::new(strategy, dataset, self.batcher, rng))
    }
}
