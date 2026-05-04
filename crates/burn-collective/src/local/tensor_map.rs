//! # Common Tensor Map for Local Collective Operations
use burn_backend::{
    Backend, TensorMetadata,
    tensor::{Device, FloatTensor},
};
use burn_std::Shape;
use std::collections::HashMap;

use crate::PeerId;

pub type CollectiveTensorMap<B> = HashMap<PeerId, FloatTensor<B>>;

pub type PeerDeviceMap<B> = HashMap<PeerId, Device<B>>;

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
