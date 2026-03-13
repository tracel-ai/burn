//! Implements the collective ring all-reduce algorithm on the global level

use core::ops::Range;
use std::{collections::HashMap, sync::Arc};

use crate::{
    NodeId,
    global::shared::GlobalCollectiveError,
    local::{get_ring_reduce_slice_ranges, get_slice_dim},
    node::sync::SyncService,
};
use burn_communication::{Address, Protocol, data_service::TensorDataService};
use burn_tensor::{Slice, TensorMetadata, backend::Backend};

// https://blog.dailydoseofds.com/p/all-reduce-and-ring-reduce-for-model

// Example: tensors=3, slices=3

// phase 1
// o->o  o
// o  o->o
//>o  o  o->

// o  1->o
//>o  o  1->
// 1->o  o

// o  1  2
// 2  o  1
// 1  2  o

// phase 2
//>o  1  2->
// 2->o  1
// 1  2->o

// 2->1  2
// 2  2->1
//>1  2  2->

// 2  2  2
// 2  2  2
// 2  2  2

/// Ring all-reduce algorithm with summation
///
/// * `node` - The id of the current node
/// * `nodes` - Map of all nodes in the operation
/// * `data_service` - The data service handles peer-to-peer tensor transfers
/// * `sync_service` - The sync service handles syncing with peers
/// * `tensor` - The tensor to reduce. At least one dimension size must be greater than the number
///   of nodes
pub(crate) async fn ring_all_reduce_sum<B, P>(
    node: NodeId,
    nodes: &HashMap<NodeId, Address>,
    data_service: Arc<TensorDataService<B, P>>,
    sync_service: Arc<SyncService<P>>,
    tensor: B::FloatTensorPrimitive,
) -> Result<B::FloatTensorPrimitive, GlobalCollectiveError>
where
    B: Backend,
    P: Protocol,
{
    let shape = tensor.shape();

    let device = &B::float_device(&tensor);
    // Slice tensors in N parts, N is node count
    let slice_dim = get_slice_dim(&shape);
    if shape[slice_dim] < nodes.len() {
        return Err(GlobalCollectiveError::RingReduceImpossible);
    }

    let ring = get_ring_topology(nodes.keys().cloned().collect::<Vec<_>>());
    let slice_ranges = get_ring_reduce_slice_ranges(shape[slice_dim], ring.len());
    let mut slices = slice_tensor::<B>(tensor, slice_dim, slice_ranges);

    let mut send_slice_idx = ring
        .iter()
        .position(|id| *id == node)
        .expect("Node is in ring");
    let prev_node_idx = (send_slice_idx + ring.len() - 1) % ring.len(); // +ring.len for overflow
    let prev_node = nodes.get(&ring[prev_node_idx]).unwrap();
    let mut transfer_counter: u64 = 0;

    // Phase 1: add
    do_cycles::<B, P>(
        &mut slices,
        &mut transfer_counter,
        &mut send_slice_idx,
        true,
        prev_node.clone(),
        &data_service,
        device,
    )
    .await?;

    // Phase 2: replace
    do_cycles::<B, P>(
        &mut slices,
        &mut transfer_counter,
        &mut send_slice_idx,
        false,
        prev_node.clone(),
        &data_service,
        device,
    )
    .await?;

    // Wait for all nodes to finish
    sync_service.sync().await;

    // merge slices
    Ok(B::float_cat(slices, slice_dim))
}

/// Do N-1 cycles of ring-reduce
///
/// * `slices` - Slices of the original tensor, len equal to node count
/// * `transfer_counter` - counter for each step (one send one receive)
/// * `send_slice_idx` - counter for the index of each slice to send
/// * `is_phase_one` - In phase 1, the tensors are aggregated. Otherwise, they are overridden
/// * `data_service` - TensorDataService for peer-to-peer tensor transfers
/// * `device` - The device on which all local tensors are stored. Should match `slices`
async fn do_cycles<B, P>(
    slices: &mut [B::FloatTensorPrimitive],
    transfer_counter: &mut u64,
    send_slice_idx: &mut usize,
    is_phase_one: bool,
    prev_node: Address,
    data_service: &Arc<TensorDataService<B, P>>,
    device: &B::Device,
) -> Result<(), GlobalCollectiveError>
where
    B: Backend,
    P: Protocol,
{
    let slice_count = slices.len();
    for _ in 0..(slice_count - 1) {
        let transfer_id = (*transfer_counter).into();
        // +slice_count to avoid overflow
        let recv_slice_idx = (*send_slice_idx + slice_count - 1) % slice_count;
        let slice_send = slices[*send_slice_idx].clone();

        let upload = {
            let data_service = data_service.clone();
            tokio::spawn(async move {
                data_service
                    .expose(slice_send.clone(), 1, transfer_id)
                    .await
            })
        };
        let download = {
            let data_client = data_service.clone();
            let next_node = prev_node.clone();
            tokio::spawn(async move { data_client.download_tensor(next_node, transfer_id).await })
        };

        upload.await.unwrap();
        let download = download.await.unwrap();
        if is_phase_one {
            let download = download.expect("Peer closed download connection");
            let tensor = B::float_from_data(download, device);
            slices[recv_slice_idx] = B::float_add(slices[recv_slice_idx].clone(), tensor);
        } else {
            let tensor = B::float_from_data(download.unwrap(), device);
            let old_shape = slices[recv_slice_idx].shape();
            if old_shape != tensor.shape() {
                return Err(GlobalCollectiveError::PeerSentIncoherentTensor);
            }
            slices[recv_slice_idx] = tensor;
        }

        // Move slice index
        *send_slice_idx = recv_slice_idx;
        *transfer_counter += 1;
    }

    Ok(())
}

/// But a tensor into even slices across a dimension
///
/// * `tensor` - the tensor to slice
/// * `slice_dim` - the dimension to slice across
/// * `slice_ranges` - The ranges of indices on `slice_dim` to use when slicing the tensor
fn slice_tensor<B: Backend>(
    tensor: B::FloatTensorPrimitive,
    slice_dim: usize,
    slice_ranges: Vec<Range<usize>>,
) -> Vec<B::FloatTensorPrimitive> {
    let shape = tensor.shape();
    // full range across all dims as Slice
    let full_range = shape
        .iter()
        .map(|dim| Slice::from(0..*dim))
        .collect::<Vec<Slice>>();

    // Slice tensors
    let mut slices = vec![];
    for range in &slice_ranges {
        let mut all_ranges = full_range.clone();
        all_ranges[slice_dim] = Slice::from(range.clone());
        let slice = B::float_slice(tensor.clone(), &all_ranges);
        slices.push(slice);
    }

    slices
}

/// Get the ring topology
fn get_ring_topology(mut nodes: Vec<NodeId>) -> Vec<NodeId> {
    // This ordering could be more sophisticated, using node proximities etc
    nodes.sort();

    nodes
}
