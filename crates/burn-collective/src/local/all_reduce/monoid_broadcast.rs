//! Monoid AllReduce

use crate::local::CollectiveTensorMap;
use crate::{PeerId, ReduceOperation};
use burn_tensor::ElementConversion;
use burn_tensor::backend::{Backend, DeviceId, DeviceOps};
use std::collections::HashMap;

#[tracing::instrument(skip(tensors))]
pub(crate) fn all_reduce_monoid_broadcast<B: Backend>(
    tensors: CollectiveTensorMap<B>,
    op: ReduceOperation,
) -> CollectiveTensorMap<B> {
    let count = tensors.len();
    assert_ne!(count, 0);
    let peer_devices = tensors
        .iter()
        .map(|(id, t)| (*id, B::float_device(t)))
        .collect::<HashMap<PeerId, B::Device>>();

    let tensors = tensors.into_values().collect::<Vec<_>>();

    let grouped: HashMap<DeviceId, Vec<B::FloatTensorPrimitive>> =
        tensors.into_iter().fold(HashMap::new(), |mut acc, t| {
            let device = B::float_device(&t);
            acc.entry(device.id()).or_default().push(t);
            acc
        });

    #[tracing::instrument(skip(tensors))]
    fn sum_device_local<B: Backend>(
        tensors: Vec<B::FloatTensorPrimitive>,
    ) -> B::FloatTensorPrimitive {
        let mut it = tensors.into_iter();
        let first = it.next().unwrap();
        it.fold(first, |a, b| B::float_add(a, b))
    }

    let tensors = grouped
        .into_values()
        .map(sum_device_local::<B>)
        .collect::<Vec<_>>();

    #[tracing::instrument(skip(tensors))]
    fn move_all<B: Backend>(
        tensors: Vec<B::FloatTensorPrimitive>,
        device: &B::Device,
    ) -> Vec<B::FloatTensorPrimitive> {
        tensors
            .into_iter()
            .map(|t| B::float_to_device(t, &device))
            .collect()
    }

    let device = B::float_device(tensors.first().unwrap());

    let tensors = move_all::<B>(tensors, &device);

    let mut source = sum_device_local::<B>(tensors);

    #[tracing::instrument(skip(tensor))]
    fn rescale<B: Backend>(
        tensor: B::FloatTensorPrimitive,
        count: usize,
    ) -> B::FloatTensorPrimitive {
        let tensor_count = count as f32;
        B::float_div_scalar(tensor, tensor_count.elem())
    }

    if op == ReduceOperation::Mean {
        source = rescale::<B>(source, count);
    }

    #[tracing::instrument(skip(source, peer_devices))]
    fn push_all<B: Backend>(
        source: B::FloatTensorPrimitive,
        peer_devices: HashMap<PeerId, B::Device>,
    ) -> CollectiveTensorMap<B> {
        let unique_devices: HashMap<DeviceId, B::Device> = peer_devices
            .values()
            .map(|device| (device.id(), device.clone()))
            .collect();

        let device_tensors: HashMap<DeviceId, B::FloatTensorPrimitive> = unique_devices
            .into_values()
            .map(|device| (device.id(), B::float_to_device(source.clone(), &device)))
            .collect();

        peer_devices
            .into_iter()
            .map(|(id, device)| (id, device_tensors.get(&device.id()).unwrap().clone()))
            .collect()
    }

    push_all::<B>(source, peer_devices)
}
