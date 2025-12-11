//! # Common Tensor Map for Local Collective Operations
use crate::PeerId;
use burn_std::Shape;
use burn_tensor::TensorMetadata;
use burn_tensor::backend::Backend;
use std::collections::HashMap;

pub type CollectiveTensorMap<B> = HashMap<PeerId, <B as Backend>::FloatTensorPrimitive>;

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
