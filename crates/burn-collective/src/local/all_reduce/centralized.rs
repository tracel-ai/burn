use std::collections::HashMap;

use burn_tensor::backend::Backend;

use crate::{
    PeerId,
    local::{broadcast_centralized, reduce_sum_centralized},
};

/// Perform an all-reduce operation by reducing all tensors on one device, and broadcasting the
/// result to all other devices
///
/// Internally, this is just a call to `reduce` followed by a `broadcast`
#[tracing::instrument(skip(tensors))]
pub(crate) fn all_reduce_sum_centralized<B: Backend>(
    tensors: &mut HashMap<PeerId, B::FloatTensorPrimitive>,
) {
    // Get corresponding devices for each peer
    let devices = tensors
        .iter()
        .map(|(id, tensor)| (*id, B::float_device(tensor)))
        .collect::<HashMap<PeerId, B::Device>>();
    let central_device = *tensors.keys().next().unwrap();

    // Reduce to central device
    let central_tensor = reduce_sum_centralized::<B>(core::mem::take(tensors), &central_device);

    // Broadcast result to all
    *tensors = broadcast_centralized::<B>(devices, central_device, central_tensor);
}
