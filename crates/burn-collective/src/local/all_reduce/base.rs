use std::collections::HashMap;

use burn_communication::websocket::WebSocket;
use burn_tensor::{ElementConversion, Shape, TensorMetadata, backend::Backend};

use crate::local::all_reduce::monoid_broadcast::all_reduce_monoid_broadcast;
use crate::{
    AllReduceStrategy, CollectiveConfig, CollectiveError, PeerId, ReduceOperation,
    local::{
        all_reduce_sum_centralized, all_reduce_sum_ring, all_reduce_sum_tree,
        broadcast_centralized, broadcast_tree, reduce_sum_centralized, reduce_sum_tree,
    },
    node::base::Node,
};

pub type CollectiveTensorMap<B> = HashMap<PeerId, <B as Backend>::FloatTensorPrimitive>;

/// Get the shape of the tensors. They should all have the same shape, otherwise None is returned.
pub(crate) fn get_common_shape<B: Backend>(tensors: &CollectiveTensorMap<B>) -> Option<Shape> {
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

/// Perform an all-reduce with no multi-node operations (global ops)
#[tracing::instrument(skip(tensors, config))]
pub(crate) async fn all_reduce_local_only<B: Backend>(
    tensors: CollectiveTensorMap<B>,
    op: ReduceOperation,
    config: &CollectiveConfig,
) -> Result<CollectiveTensorMap<B>, CollectiveError> {
    let local_strategy = &config.local_all_reduce_strategy;

    if false {
        let mut tensors = match local_strategy {
            AllReduceStrategy::Centralized => all_reduce_sum_centralized::<B>(tensors),
            AllReduceStrategy::Tree(arity) => all_reduce_sum_tree::<B>(tensors, *arity),
            AllReduceStrategy::Ring => all_reduce_sum_ring::<B>(tensors),
        };

        if op == ReduceOperation::Mean {
            // Apply mean division
            let tensor_count = tensors.len() as f32;
            tensors.iter_mut().for_each(|(_, tensor)| {
                *tensor = B::float_div_scalar(tensor.clone(), tensor_count.elem())
            });
        }
        Ok(tensors)
    } else {
        Ok(all_reduce_monoid_broadcast::<B>(tensors, op))
    }
}

/// Do an all-reduce in a multi-node context
///
/// With Tree and Centralized strategies, the all-reduce is split between a
/// reduce (all tensors are reduced to one device), and a broadcast (the result is sent to all
/// other devices). The all-reduce on the global level is done between both steps.
/// Due to the nature of the Ring strategy, this separation can't be done.
// For the Ring strategy, this isn't possible, because it is more like a
// reduce-scatter plus an all-gather, so using a Ring strategy locally in a multi-node
// setup may be unadvantageous.
#[tracing::instrument(skip(tensors, config, global_client))]
pub(crate) async fn all_reduce_with_global<B: Backend>(
    tensors: CollectiveTensorMap<B>,
    op: ReduceOperation,
    config: &CollectiveConfig,
    global_client: &mut Node<B, WebSocket>,
) -> Result<CollectiveTensorMap<B>, CollectiveError> {
    let local_strategy = config.local_all_reduce_strategy;
    let global_strategy = config.global_all_reduce_strategy;

    // Get corresponding devices for each peer
    let devices = tensors
        .iter()
        .map(|(id, tensor)| (*id, B::float_device(tensor)))
        .collect::<HashMap<PeerId, B::Device>>();

    // For Centralized and Tree, we only need to do a reduce here, we'll do a broadcast later
    let main_device = *tensors.keys().next().unwrap();

    let mut main_tensor = match local_strategy {
        AllReduceStrategy::Centralized => reduce_sum_centralized::<B>(tensors, &main_device),
        AllReduceStrategy::Tree(arity) => reduce_sum_tree::<B>(tensors, &main_device, arity),
        AllReduceStrategy::Ring => all_reduce_sum_ring::<B>(tensors)
            .remove(&main_device)
            .unwrap(),
    };

    // Do aggregation on global level with the main tensor
    main_tensor = global_client
        .all_reduce(main_tensor, global_strategy.unwrap(), op)
        .await
        .map_err(CollectiveError::Global)?;

    // Broadcast result to all devices
    let tensors = match local_strategy {
        AllReduceStrategy::Tree(arity) => {
            broadcast_tree::<B>(devices, main_device, main_tensor, arity)
        }
        // If we chose the ring strategy and we must still broadcast the global result,
        // we use the centralized strategy for broadcasting, but the tree may be better.
        AllReduceStrategy::Centralized | AllReduceStrategy::Ring => {
            broadcast_centralized::<B>(devices, main_device, main_tensor)
        }
    };

    Ok(tensors)
}
