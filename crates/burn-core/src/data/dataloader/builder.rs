use super::{
    BatchDataLoader, BatchStrategy, DataLoader, FixBatchStrategy, MultiThreadDataLoader,
    batcher::Batcher,
};
use burn_dataset::Dataset;
use burn_tensor::backend::Backend;
use rand::{SeedableRng, rngs::StdRng};
use std::sync::Arc;

/// A builder for data loaders.
pub struct DataLoaderBuilder<B: Backend, I, O> {
    strategy: Option<Box<dyn BatchStrategy<I>>>,
    batcher: Arc<dyn Batcher<B, I, O>>,
    num_threads: Option<usize>,
    shuffle: Option<u64>,
    device: Option<B::Device>,
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
        Bt: Batcher<B, I, O> + 'static,
    {
        Self {
            batcher: Arc::new(batcher),
            strategy: None,
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
    pub fn set_device(mut self, device: B::Device) -> Self {
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
    pub fn build<D>(self, dataset: D) -> Arc<dyn DataLoader<B, O>>
    where
        D: Dataset<I> + 'static,
    {
        let dataset = Arc::new(dataset);

        let device = self.device.unwrap_or_default();
        let rng = self.shuffle.map(StdRng::seed_from_u64);
        let strategy = match self.strategy {
            Some(strategy) => strategy,
            None => Box::new(FixBatchStrategy::new(1)),
        };

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
    use super::*;
    use crate::TestBackend;
    use crate::data::dataset::FakeDataset;

    #[derive(new, Clone)]
    struct TestBatcherDevice;

    #[cfg(test)]
    impl<I> Batcher<TestBackend, I, TestDevice> for TestBatcherDevice {
        fn batch(&self, _items: Vec<I>, device: &TestDevice) -> TestDevice {
            *device
        }
    }

    type TestDevice = <TestBackend as Backend>::Device;

    #[test]
    fn test_dataloader_no_workers() {
        type TestDevice = <TestBackend as Backend>::Device;

        let default_device = TestDevice::default();
        let dataloader = DataLoaderBuilder::new(TestBatcherDevice::new())
            .batch_size(1)
            .build(FakeDataset::<String>::new(9));

        assert_eq!(dataloader.num_items(), 9);

        for device in dataloader.iter() {
            assert_eq!(device, default_device)
        }
    }

    #[test]
    fn test_dataloader_default_device() {
        let default_device = TestDevice::default();
        let dataloader = DataLoaderBuilder::new(TestBatcherDevice::new())
            .batch_size(1)
            .num_workers(1)
            .build(FakeDataset::<String>::new(9));

        assert_eq!(dataloader.num_items(), 9);

        for device in dataloader.iter() {
            assert_eq!(device, default_device)
        }
    }

    #[test]
    fn test_dataloader_slice_multi_device() {
        let dataloader = DataLoaderBuilder::new(TestBatcherDevice::new())
            .batch_size(1)
            .num_workers(1)
            .build(FakeDataset::<String>::new(11));

        #[cfg(all(
            test,
            not(feature = "test-tch"),
            not(feature = "test-wgpu"),
            not(feature = "test-cuda")
        ))]
        // Only one device exists...
        let (device1, device2) = (
            burn_ndarray::NdArrayDevice::Cpu,
            burn_ndarray::NdArrayDevice::Cpu,
        );

        #[cfg(all(test, feature = "test-tch"))]
        let (device1, device2) = (
            burn_tch::LibTorchDevice::Cuda(0),
            burn_tch::LibTorchDevice::Cuda(1),
        );

        #[cfg(all(test, feature = "test-wgpu"))]
        let (device1, device2) = (
            burn_wgpu::WgpuDevice::DiscreteGpu(0),
            burn_wgpu::WgpuDevice::DiscreteGpu(1),
        );

        #[cfg(all(test, feature = "test-cuda"))]
        let (device1, device2) = (burn_cuda::CudaDevice::new(0), burn_cuda::CudaDevice::new(1));

        assert_eq!(dataloader.num_items(), 11);
        let dataloader_1 = dataloader.slice(0, 5).to_device(&device1);
        let dataloader_2 = dataloader.slice(5, 11).to_device(&device2);

        assert_eq!(dataloader_1.num_items(), 5);
        assert_eq!(dataloader_2.num_items(), 6);

        let (mut iterator_1, mut iterator_2) = (dataloader_1.iter(), dataloader_2.iter());

        for _ in 0..5 {
            assert_eq!(iterator_1.next(), Some(device1));
            assert_eq!(iterator_2.next(), Some(device2));
        }

        assert_eq!(iterator_1.next(), None);
        // For uneven split, the last dataloader (partial dataset) will have the remaining item
        assert_eq!(iterator_2.next(), Some(device2));
        assert_eq!(iterator_2.next(), None);
    }
}
