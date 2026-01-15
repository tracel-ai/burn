use crate::local::tensor_map::{CollectiveTensorMap, get_peer_devices};
use crate::{
    AllReduceStrategy, CollectiveConfig, CollectiveError, ReduceOperation,
    local::{
        all_reduce_sum_centralized, all_reduce_sum_ring, all_reduce_sum_tree,
        broadcast_centralized, broadcast_tree, reduce_sum_centralized, reduce_sum_tree,
    },
    node::base::Node,
};
use burn_communication::Protocol;
use burn_tensor::{ElementConversion, backend::Backend};

#[cfg(feature = "tracing")]
use tracing::Instrument;

/// Perform an all-reduce with no multi-node operations (global ops)
#[cfg_attr(
    feature = "tracing",
    tracing::instrument(level = "trace", skip(tensors, config))
)]
pub(crate) async fn all_reduce_local_only<B: Backend>(
    tensors: CollectiveTensorMap<B>,
    op: ReduceOperation,
    config: &CollectiveConfig,
) -> Result<CollectiveTensorMap<B>, CollectiveError> {
    let local_strategy = &config.local_all_reduce_strategy;

    let mut reduced_tensors = match local_strategy {
        AllReduceStrategy::Centralized => all_reduce_sum_centralized::<B>(tensors),
        AllReduceStrategy::Tree(arity) => all_reduce_sum_tree::<B>(tensors, *arity),
        AllReduceStrategy::Ring => all_reduce_sum_ring::<B>(tensors),
    };

    if op == ReduceOperation::Mean {
        #[cfg(feature = "tracing")]
        let _span = tracing::info_span!("mean_reduction").entered();

        // Apply mean division
        let div = (reduced_tensors.len() as f32).elem();

        reduced_tensors = reduced_tensors
            .into_iter()
            .map(|(id, t)| (id, B::float_div_scalar(t, div)))
            .collect();
    }
    Ok(reduced_tensors)
}

/// Do an all-reduce in a multi-node context
///
/// With Tree and Centralized strategies, the all-reduce is split between a
/// reduce (all tensors are reduced to one device), and a broadcast (the result is sent to all
/// other devices). The all-reduce on the global level is done between both steps.
/// Due to the nature of the Ring strategy, this separation can't be done.
///
/// For the Ring strategy, this isn't possible, because it is more like a
/// reduce-scatter plus an all-gather, so using a Ring strategy locally in a multi-node
/// setup may be unadvantageous.
#[cfg_attr(
    feature = "tracing",
    tracing::instrument(level = "trace", skip(tensors, config, global_client))
)]
pub(crate) async fn all_reduce_with_global<B: Backend, P: Protocol>(
    tensors: CollectiveTensorMap<B>,
    op: ReduceOperation,
    config: &CollectiveConfig,
    global_client: &mut Node<B, P>,
) -> Result<CollectiveTensorMap<B>, CollectiveError> {
    let peer_devices = get_peer_devices::<B>(&tensors);

    // For Centralized and Tree, we only need to do a reduce here, we'll do a broadcast later
    let main_device = *tensors.keys().next().unwrap();

    let mut main_tensor = match config.local_all_reduce_strategy {
        AllReduceStrategy::Centralized => reduce_sum_centralized::<B>(tensors, &main_device),
        AllReduceStrategy::Tree(arity) => reduce_sum_tree::<B>(tensors, &main_device, arity),
        AllReduceStrategy::Ring => all_reduce_sum_ring::<B>(tensors)
            .remove(&main_device)
            .unwrap(),
    };

    // Do aggregation on global level with the main tensor
    main_tensor = {
        let fut = async {
            let global_strategy = config
                .global_all_reduce_strategy
                .expect("global_all_reduce_strategy must be set");

            global_client
                .all_reduce(main_tensor, global_strategy, op)
                .await
        };
        #[cfg(feature = "tracing")]
        {
            fut.instrument(tracing::info_span!("global_all_reduce"))
        }
        #[cfg(not(feature = "tracing"))]
        {
            fut
        }
    }
    .await
    .map_err(CollectiveError::Global)?;

    // Broadcast result to all devices
    let tensors = match config.local_all_reduce_strategy {
        AllReduceStrategy::Tree(arity) => {
            broadcast_tree::<B>(peer_devices, main_device, main_tensor, arity)
        }
        // If we chose the ring strategy and we must still broadcast the global result,
        // we use the centralized strategy for broadcasting, but the tree may be better.
        AllReduceStrategy::Centralized | AllReduceStrategy::Ring => {
            broadcast_centralized::<B>(peer_devices, main_device, main_tensor)
        }
    };

    Ok(tensors)
}
