use super::{
    batcher::DynBatcher, BatchDataLoader, BatchStrategy, DataLoader, DistributionStrategy,
    FixBatchStrategy, FixedDistributor, RoundRobinDistributor,
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
    devices: Option<Vec<B::Device>>,
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
    pub fn new<Ba>(batcher: Ba) -> Self
    where
        Ba: DynBatcher<B, I, O> + 'static,
    {
        Self {
            batcher: Box::new(batcher),
            strategy: None,
            num_threads: None,
            shuffle: None,
            devices: None,
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

    /// Sets the data loader devices.
    ///
    /// # Arguments
    ///
    /// * `devices` - The devices to use when loading batches.
    ///
    /// # Returns
    ///
    /// The data loader builder.
    pub fn devices(mut self, devices: Vec<B::Device>) -> Self {
        self.devices = Some(devices);
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

        let devices = self.devices.unwrap_or(vec![Default::default()]);
        let num_devices = devices.len();

        // NOTE: maybe this could be configurable?
        let distributor: Box<dyn DistributionStrategy<Resource = B::Device>> = if num_devices > 1 {
            // Round-robin device selection to alternate each batch when multiple GPUs are used.
            // This way, each batch should already be assigned to the correct device before being
            // passed to the model.
            Box::new(RoundRobinDistributor::new(devices.clone()))
        } else {
            Box::new(FixedDistributor::new(devices.clone()))
        };

        if let Some(num_threads) = self.num_threads {
            // For multi-device and multi-thread, each thread is responsible to load the data on a specific device.
            // This should be good enough to balance the batches between the devices, and the main data loader still
            // uses the round-robin strategy to make sure each batch alternates to match the multi-device training.
            let distributors = if num_devices > 1 && num_threads > 1 {
                Some(
                    (0..num_threads)
                        .map(|i| {
                            let device_id = i % num_devices;
                            let distributor: Box<dyn DistributionStrategy<Resource = B::Device>> =
                                Box::new(
                                    FixedDistributor::new(devices.clone()).with_fixed(device_id),
                                );
                            distributor
                        })
                        .collect::<Vec<_>>(),
                )
            } else {
                None
            };

            return Arc::new(BatchDataLoader::multi_thread(
                strategy,
                dataset,
                self.batcher,
                num_threads,
                distributor,
                distributors,
                rng,
            ));
        }

        Arc::new(BatchDataLoader::new(
            strategy,
            dataset,
            self.batcher,
            distributor,
            rng,
        ))
    }
}
