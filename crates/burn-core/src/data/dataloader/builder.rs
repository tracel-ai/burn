use super::{
    batcher::DynBatcher, BatchDispatcher, BatchStrategy, FixBatchStrategy, LazyBatchDataLoader,
    LazyDataLoader,
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
    dispatcher: Option<Box<dyn BatchDispatcher<Resource = B::Device>>>,
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
            dispatcher: None,
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

    /// Sets the data loader device dispatching/selection strategy for a batch.
    ///
    /// # Arguments
    ///
    /// * `dispatcher` - The device dispatching strategy.
    ///
    /// # Returns
    ///
    /// The data loader builder.
    pub fn dispatcher<D>(mut self, dispatcher: D) -> Self
    where
        D: BatchDispatcher<Resource = B::Device>,
    {
        self.dispatcher = Some(dispatcher.clone_dyn());
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
            self.dispatcher,
            rng,
            self.num_threads.unwrap_or(0),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::dataloader::FixedDispatcher;
    use crate::data::dataset::FakeDataset;
    use crate::{data::dataloader::batcher::Batcher, TestBackend};

    #[test]
    fn test_default_device_distributor() {
        type TestDevice = <TestBackend as Backend>::Device;

        #[derive(new, Clone)]
        pub struct TestBatcher;

        #[cfg(test)]
        impl<I> Batcher<TestBackend, I, TestDevice> for TestBatcher {
            fn batch(&self, _items: Vec<I>, device: &TestDevice) -> TestDevice {
                device.clone()
            }
        }

        // `LazyBatchDataLoader` with no BatchDispatcher fixed device (default)
        let default_device = TestDevice::default();
        let dataloader = DataLoaderBuilder::new(TestBatcher::new())
            .batch_size(1)
            .num_workers(1)
            .build(FakeDataset::<String>::new(9));

        for device in dataloader.iter() {
            assert_eq!(device, default_device)
        }
    }

    #[test]
    fn test_multi_device_distributor() {
        type TestDevice = <TestBackend as Backend>::Device;

        #[derive(new, Clone)]
        pub struct TestBatcher;

        #[cfg(test)]
        impl<I> Batcher<TestBackend, I, TestDevice> for TestBatcher {
            fn batch(&self, _items: Vec<I>, device: &TestDevice) -> TestDevice {
                device.clone()
            }
        }

        // `LazyBatchDataLoader` with no BatchDispatcher fixed device (default)
        let num_items = 11;
        let dataloader = DataLoaderBuilder::new(TestBatcher::new())
            .batch_size(1)
            .num_workers(1)
            .build(FakeDataset::<String>::new(num_items));

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

        let fixed_devices = vec![
            FixedDispatcher::new(vec![device1.clone()]).clone_dyn(),
            FixedDispatcher::new(vec![device2.clone()]).clone_dyn(),
        ];
        let dataloaders = dataloader.split(fixed_devices);

        assert_eq!(dataloaders.len(), 2);

        let (mut iterator_1, mut iterator_2) = (dataloaders[0].iter(), dataloaders[1].iter());

        for _ in 0..num_items / 2 {
            assert_eq!(iterator_1.next(), Some(device1.clone()));
            assert_eq!(iterator_2.next(), Some(device2.clone()));
        }

        assert_eq!(iterator_1.next(), None);
        // For uneven split, the last dataloader (partial dataset) will have the remaining item
        assert_eq!(iterator_2.next(), Some(device2));
        assert_eq!(iterator_2.next(), None);
    }
}
