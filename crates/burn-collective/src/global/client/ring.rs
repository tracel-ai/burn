use std::sync::Arc;

use burn_network::network::{NetworkClient, NetworkServer};
use burn_tensor::backend::Backend;
use crate::global::{client::data_server::{TensorDataClient, TensorDataService}, shared::base::RingAllReduceStrategy};


pub(crate) async fn ring_all_reduce<B, C, S>(
    _data_client: &TensorDataClient<B, C, S>,
    _tensor: B::FloatTensorPrimitive,
    _device: &B::Device,
    _strategy: RingAllReduceStrategy,
) -> B::FloatTensorPrimitive 
where 
    B: Backend, 
    C: NetworkClient,
    S: NetworkServer<State = Arc<TensorDataService<B, C>>>
{
    // Slice the tensor, should correspond to the local slicing.
    todo!()
}

