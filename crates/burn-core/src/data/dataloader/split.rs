use std::sync::Arc;

use burn_tensor::backend::Backend;

use super::DataLoader;

/// Splits a dataloader into multiple partial dataloaders (one per device).
pub fn split_dataloader<B: Backend, O>(
    dataloader: Arc<dyn DataLoader<B, O>>,
    devices: &[B::Device],
) -> Vec<Arc<dyn DataLoader<B, O>>> {
    let num_splits = devices.len();
    if num_splits > 1 {
        let num_items = dataloader.num_items();
        let mut dataloaders = Vec::with_capacity(num_splits);

        let mut start = 0;
        let step = num_items / num_splits;
        for (i, device) in devices.iter().enumerate() {
            let end = if i == (num_splits - 1) {
                num_items
            } else {
                start + step
            };
            let mut dataloader = dataloader.slice(start, end);
            dataloader.set_device(device.clone());
            dataloaders.push(Arc::from(dataloader));
            start = end;
        }
        dataloaders
    } else {
        vec![dataloader]
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::TestBackend;
    use crate::data::dataloader::batcher::Batcher;
    use crate::data::dataloader::{BatchDataLoader, FixBatchStrategy};
    use crate::data::dataset::FakeDataset;

    #[test]
    fn test_split_batch_dataloader() {
        type TestDevice = <TestBackend as Backend>::Device;

        #[derive(new, Clone)]
        pub struct TestBatcher;

        #[cfg(test)]
        impl<I> Batcher<TestBackend, I, (Vec<I>, TestDevice)> for TestBatcher {
            fn batch(&self, items: Vec<I>, device: &TestDevice) -> (Vec<I>, TestDevice) {
                (items, *device)
            }
        }

        let batcher = Box::new(TestBatcher::new());
        let dataset = Arc::new(FakeDataset::<String>::new(11));
        let dataloader = Arc::new(BatchDataLoader::new(
            Box::new(FixBatchStrategy::new(5)),
            dataset.clone(),
            batcher,
            Default::default(),
            None,
        ));

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

        let dataloaders = split_dataloader(dataloader.clone(), &[device1, device2]);

        assert_eq!(dataloaders.len(), 2);

        let [dataloader_1, dataloader_2] = match dataloaders.try_into() {
            Ok(arr) => arr,
            Err(_) => unreachable!(),
        };
        assert_eq!(dataloader_1.num_items(), 5);
        assert_eq!(dataloader_2.num_items(), 6);

        let mut items_dataloader = HashSet::new();
        let mut items_dataloader_split = HashSet::new();

        for (items, _device) in dataloader.iter() {
            for item in items {
                items_dataloader.insert(item);
            }
        }

        for (items, device) in dataloader_1.iter() {
            assert_eq!(device, device1);
            for item in items {
                items_dataloader_split.insert(item);
            }
        }

        for (items, device) in dataloader_2.iter() {
            assert_eq!(device, device2);
            for item in items {
                items_dataloader_split.insert(item);
            }
        }

        assert_eq!(items_dataloader, items_dataloader_split);
    }
}
