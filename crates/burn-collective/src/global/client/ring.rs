use core::ops::Range;
use std::sync::Arc;

use crate::global::shared::base::RingAllReduceStrategy;
use burn_network::{
    data_service::TensorDataService,
    network::{Network, NetworkAddress},
};
use burn_tensor::{TensorMetadata, backend::Backend};

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
pub(crate) async fn ring_all_reduce_sum<B, N>(
    data_service: Arc<TensorDataService<B, N>>,
    tensor: B::FloatTensorPrimitive,
    device: &B::Device,
    strategy: RingAllReduceStrategy,
) -> B::FloatTensorPrimitive
where
    B: Backend,
    N: Network,
{
    let mut slices = slice_tensor::<B>(tensor, strategy.slice_dim, strategy.slice_ranges);
    let mut send_slice_idx = strategy.first_slice;
    let mut transfer_counter: u32 = 0;

    // Phase 1: add
    do_cycles::<B, N>(
        &mut slices,
        &mut transfer_counter,
        &mut send_slice_idx,
        true,
        strategy.next_node.clone(),
        &data_service,
        device,
    )
    .await;

    // Phase 2: replace
    do_cycles::<B, N>(
        &mut slices,
        &mut transfer_counter,
        &mut send_slice_idx,
        false,
        strategy.next_node,
        &data_service,
        device,
    )
    .await;

    // merge slices
    B::float_cat(slices, strategy.slice_dim)
}

/// Do N-1 cycles of ring-reduce
async fn do_cycles<B, N>(
    slices: &mut [B::FloatTensorPrimitive],
    transfer_counter: &mut u32,
    send_slice_idx: &mut usize,
    is_phase_one: bool,
    next_node: NetworkAddress,
    data_service: &Arc<TensorDataService<B, N>>,
    device: &B::Device,
) where
    B: Backend,
    N: Network,
{
    let slice_count = slices.len();
    for _ in 0..(slice_count - 1) {
        let transfer_id = (*transfer_counter).into();
        let recv_slice_idx = (*send_slice_idx - 1) % slice_count;
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
            let next_node = next_node.clone();
            tokio::spawn(async move { data_client.download_tensor(next_node, transfer_id).await })
        };

        upload.await.unwrap();
        let download = download.await.unwrap();
        if is_phase_one {
            let tensor = B::float_from_data(download.unwrap(), device);
            slices[recv_slice_idx] = B::float_add(slices[recv_slice_idx].clone(), tensor);
        } else {
            // TODO check dimensions of new slice
            slices[recv_slice_idx] = B::float_from_data(download.unwrap(), device);
        }

        // Move slice index
        *send_slice_idx = (*send_slice_idx - 1) % slice_count;
        *transfer_counter += 1;
    }
}

fn slice_tensor<B: Backend>(
    tensor: B::FloatTensorPrimitive,
    slice_dim: usize,
    slice_ranges: Vec<Range<usize>>,
) -> Vec<B::FloatTensorPrimitive> {
    let shape = tensor.shape();
    // full range across all dims
    let full_range = shape
        .dims
        .iter()
        .map(|dim| Range {
            start: 0,
            end: *dim,
        })
        .collect::<Vec<Range<usize>>>();

    // Get ranges for each slice across each dim (unsliced dims have full range)
    let ranges = slice_ranges
        .iter()
        .map(|r| {
            let mut range = full_range.clone();
            range[slice_dim] = r.clone();
            range
        })
        .collect::<Vec<Vec<Range<usize>>>>();

    // Slice tensors
    let mut slices = vec![];
    for range in &ranges {
        let slice = B::float_slice(tensor.clone(), range);
        slices.push(slice);
    }

    slices
}
