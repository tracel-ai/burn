use super::{
    BatchDataLoader, BatchStrategy, DataLoader, FixBatchStrategy, MultiThreadDataLoader,
    batcher::Batcher,
};
use burn_dataset::Dataset;
use burn_tensor::Device;
use rand::{SeedableRng, rngs::StdRng};
use std::sync::Arc;

/// A builder for data loaders.
pub struct DataLoaderBuilder<I, O> {
    batch_size: Option<usize>,
    drop_last: bool,
    batcher: Arc<dyn Batcher<I, O>>,
    num_threads: Option<usize>,
    shuffle: Option<u64>,
    device: Option<Device>,
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
    pub fn new<Bt>(batcher: Bt) -> Self
    where
        Bt: Batcher<I, O> + 'static,
    {
        Self {
            batcher: Arc::new(batcher),
            batch_size: None,
            drop_last: false,
            num_threads: None,
            shuffle: None,
            device: None,
        }
    }

    /// Sets the batch size to a fix number.
    ///
    /// The [fix batch strategy](FixBatchStrategy) will be used.
    ///
    /// # Arguments
    ///
    /// * `batch_size` - The batch size.
    ///
    /// # Returns
    ///
    /// The data loader builder.
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    /// Sets whether to drop the last batch when it is smaller than the batch size.
    ///
    /// This happens when the number of items is not a multiple of the batch size. By default,
    /// the last incomplete batch is kept.
    ///
    /// # Arguments
    ///
    /// * `drop_last` - Whether to drop the last incomplete batch.
    ///
    /// # Returns
    ///
    /// The data loader builder.
    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
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
    /// - `Some(0)` or `None`: the dataloader will run without work threads.
    /// - `Some(n); n > 0`: the dataloader will run with `n` background threads.
    ///
    /// A 1-worker threaded dataloader will run loads in a background thread,
    /// while a 0-worker threaded dataloader will run loads in the main thread.
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

    /// Sets the data loader device.
    ///
    /// # Arguments
    ///
    /// * `device` - The device to use when loading a batch.
    ///
    /// # Returns
    ///
    /// The data loader builder.
    pub fn set_device(mut self, device: Device) -> Self {
        self.device = Some(device);
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

        let device = self.device.unwrap_or_default();
        let rng = self.shuffle.map(StdRng::seed_from_u64);
        let strategy: Box<dyn BatchStrategy<I>> = Box::new(
            FixBatchStrategy::new(self.batch_size.unwrap_or(1)).with_drop_last(self.drop_last),
        );

        if let Some(num_threads) = self.num_threads
            && num_threads > 0
        {
            return Arc::new(MultiThreadDataLoader::new(
                strategy,
                dataset,
                self.batcher,
                num_threads,
                device,
                rng,
            ));
        }

        Arc::new(BatchDataLoader::new(
            strategy,
            dataset,
            self.batcher,
            device,
            rng,
        ))
    }
}

#[cfg(test)]
mod tests {
    #[cfg(test)]
    use burn_tensor::Device;

    use super::*;
    use crate::data::dataset::FakeDataset;

    #[derive(new, Clone)]
    struct TestBatcherDevice;

    #[cfg(test)]
    impl<I> Batcher<I, Device> for TestBatcherDevice {
        fn batch(&self, _items: Vec<I>, device: &Device) -> Device {
            device.clone()
        }
    }

    #[derive(Clone)]
    struct TestBatcherLen;

    impl<I> Batcher<I, usize> for TestBatcherLen {
        fn batch(&self, items: Vec<I>, _device: &Device) -> usize {
            items.len()
        }
    }

    #[test]
    fn test_dataloader_no_workers() {
        let default_device = Device::default();
        let dataloader = DataLoaderBuilder::new(TestBatcherDevice::new())
            .batch_size(1)
            .build(FakeDataset::<String>::new(9));

        assert_eq!(dataloader.num_items(), 9);

        for device in dataloader.iter().map(Result::unwrap) {
            assert_eq!(device, default_device)
        }
    }

    #[test]
    fn test_dataloader_drop_last() {
        for num_workers in [0, 1, 2] {
            let dataloader = DataLoaderBuilder::new(TestBatcherLen)
                .batch_size(4)
                .drop_last(true)
                .num_workers(num_workers)
                .build(FakeDataset::<String>::new(10));

            let lengths: Vec<usize> = dataloader.iter().map(Result::unwrap).collect();

            assert_eq!(lengths.iter().sum::<usize>(), 8, "workers: {num_workers}");
            assert!(
                lengths.iter().all(|len| *len == 4),
                "workers: {num_workers}"
            );
        }
    }

    #[test]
    fn test_dataloader_keeps_last_batch_by_default() {
        let dataloader = DataLoaderBuilder::new(TestBatcherLen)
            .batch_size(4)
            .build(FakeDataset::<String>::new(10));

        let lengths: Vec<usize> = dataloader.iter().map(Result::unwrap).collect();

        assert_eq!(lengths, vec![4, 4, 2]);
    }

    #[test]
    fn test_dataloader_drop_last_is_independent_of_call_order() {
        let dataloader = DataLoaderBuilder::new(TestBatcherLen)
            .drop_last(true)
            .batch_size(4)
            .build(FakeDataset::<String>::new(10));

        let lengths: Vec<usize> = dataloader.iter().map(Result::unwrap).collect();

        assert_eq!(lengths, vec![4, 4]);
    }

    #[test]
    fn test_dataloader_default_device() {
        let default_device = Device::default();
        let dataloader = DataLoaderBuilder::new(TestBatcherDevice::new())
            .batch_size(1)
            .num_workers(1)
            .build(FakeDataset::<String>::new(9));

        assert_eq!(dataloader.num_items(), 9);

        for device in dataloader.iter().map(Result::unwrap) {
            assert_eq!(device, default_device)
        }
    }

    #[test]
    fn test_dataloader_slice_multi_device() {
        let dataloader = DataLoaderBuilder::new(TestBatcherDevice::new())
            .batch_size(1)
            .num_workers(1)
            .build(FakeDataset::<String>::new(11));

        #[cfg(all(test, not(feature = "tch"), not(feature = "cuda")))]
        // Only one device exists...
        let (device1, device2) = (Device::flex(), Device::flex());

        #[cfg(all(test, feature = "tch"))]
        let (device1, device2) = (Device::libtorch_cuda(0), Device::libtorch_cuda(1));

        #[cfg(all(test, feature = "cuda"))]
        let (device1, device2) = (Device::cuda(0), Device::cuda(1));

        assert_eq!(dataloader.num_items(), 11);
        let dataloader_1 = dataloader.slice(0, 5).to_device(&device1);
        let dataloader_2 = dataloader.slice(5, 11).to_device(&device2);

        assert_eq!(dataloader_1.num_items(), 5);
        assert_eq!(dataloader_2.num_items(), 6);

        let (mut iterator_1, mut iterator_2) = (dataloader_1.iter(), dataloader_2.iter());

        for _ in 0..5 {
            assert_eq!(iterator_1.next().unwrap().unwrap(), device1);
            assert_eq!(iterator_2.next().unwrap().unwrap(), device2);
        }

        assert!(iterator_1.next().is_none());
        // For uneven split, the last dataloader (partial dataset) will have the remaining item
        assert_eq!(iterator_2.next().unwrap().unwrap(), device2);
        assert!(iterator_2.next().is_none());
    }
}
