use std::collections::HashMap;

use burn_tensor::backend::Backend;

use crate::local::tensor_map::{CollectiveTensorMap, get_peer_devices};
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
    tensors: CollectiveTensorMap<B>,
) -> HashMap<PeerId, B::FloatTensorPrimitive> {
    // Get corresponding devices for each peer
    let peer_devices = get_peer_devices::<B>(&tensors);
    let central_device = *tensors.keys().next().unwrap();

    // Reduce to central device
    let central_tensor = reduce_sum_centralized::<B>(tensors, &central_device);

    // Broadcast result to all
    broadcast_centralized::<B>(peer_devices, central_device, central_tensor)
}
