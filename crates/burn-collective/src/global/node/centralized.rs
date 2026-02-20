use std::{collections::HashMap, sync::Arc};

use crate::{NodeId, global::shared::GlobalCollectiveError, node::sync::SyncService};
use burn_communication::data_service::TensorDataService;
use burn_communication::{Address, Protocol};
use burn_tensor::TensorMetadata;
use burn_tensor::backend::Backend;
use futures::StreamExt;
use futures::stream::FuturesUnordered;

/// Global all-reduce, using a centralized strategy.
///
/// Returns the resulting tensor on the same device as the input tensor
pub(crate) async fn centralized_all_reduce_sum<B, P>(
    node: NodeId,
    nodes: &HashMap<NodeId, Address>,
    data_service: &Arc<TensorDataService<B, P>>,
    sync_service: Arc<SyncService<P>>,
    tensor: B::FloatTensorPrimitive,
    base_id: u64,
) -> Result<B::FloatTensorPrimitive, GlobalCollectiveError>
where
    B: Backend,
    P: Protocol,
{
    let ids = nodes.keys().cloned().collect::<Vec<_>>();
    let central = get_central_node(ids.clone());

    let shape = tensor.shape();
    let device = &B::float_device(&tensor);

    let res = if central == node {
        // Transfer 1: download tensors from other nodes
        let mut futures = ids
            .iter()
            .filter(|id| **id != central) // Only non-central nodes
            .map(|id| {
                let address = nodes.get(id).unwrap();
                let device = device.clone();
                let data_service = data_service.clone();
                async move {
                    let data = data_service
                        .download_tensor((*address).clone(), base_id.into())
                        .await
                        .unwrap_or_else(|| {
                            panic!("Couldn't find the tensor for transfer id {base_id}")
                        });
                    B::float_from_data(data, &device)
                }
            })
            .collect::<FuturesUnordered<_>>();

        // Sum all downloads async
        let mut sum = tensor;
        while let Some(res) = futures.next().await {
            if shape != res.shape() {
                return Err(GlobalCollectiveError::PeerSentIncoherentTensor);
            }
            sum = B::float_add(sum, res);
        }

        // Transfer 2: Expose result
        let other_nodes_count = ids.len() as u32 - 1;
        data_service
            .expose(sum.clone(), other_nodes_count, (base_id + 1).into())
            .await;

        sum
    } else {
        // Transfer 1: Expose input
        data_service.expose(tensor, 1, base_id.into()).await;

        // Transfer 2: Download result
        let central_addr = nodes.get(&central).unwrap().clone();
        let data = data_service
            .download_tensor(central_addr, (base_id + 1).into())
            .await
            .unwrap_or_else(|| panic!("Couldn't find the tensor for transfer id {}", base_id + 1));

        let res = B::float_from_data(data, device);
        if shape != res.shape() {
            return Err(GlobalCollectiveError::PeerSentIncoherentTensor);
        }

        res
    };

    // Wait for all nodes to finish
    sync_service.sync().await;

    Ok(res)
}

/// Get the central node for a centralized all-reduce
pub(crate) fn get_central_node(mut nodes: Vec<NodeId>) -> NodeId {
    nodes.sort();

    *nodes.first().unwrap()
}
