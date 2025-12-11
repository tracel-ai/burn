//! # Common Tensor Map for Local Collective Operations
use crate::PeerId;
use burn_std::Shape;
use burn_tensor::TensorMetadata;
use burn_tensor::backend::Backend;
use std::collections::HashMap;

pub type CollectiveTensorMap<B> = HashMap<PeerId, <B as Backend>::FloatTensorPrimitive>;

pub type PeerDeviceMap<B> = HashMap<PeerId, <B as Backend>::Device>;

/// Get the shape of the tensors. They should all have the same shape, otherwise None is returned.
pub fn get_common_shape<B: Backend>(tensors: &CollectiveTensorMap<B>) -> Option<Shape> {
    let mut it = tensors.values();
    if let Some(first) = it.next() {
        let shape = first.shape();
        for tensor in it {
            if tensor.shape() != shape {
                return None;
            }
        }
        return Some(shape);
    }
    None
}

/// Get the `{ peer_id -> device }` mapping for the given tensors.
pub fn get_peer_devices<B: Backend>(tensors: &CollectiveTensorMap<B>) -> PeerDeviceMap<B> {
    tensors
        .iter()
        .map(|(id, tensor)| (*id, B::float_device(tensor)))
        .collect()
}
