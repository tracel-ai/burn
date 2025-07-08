use std::sync::Arc;

use crate::global::shared::base::CentralizedAllReduceStrategy::{Central, Peripheral};
use crate::global::{
    client::data_server::{TensorDataClient, TensorDataService},
    shared::base::CentralizedAllReduceStrategy,
};
use burn_network::network::{NetworkClient, NetworkServer};
use burn_tensor::backend::Backend;
use futures::StreamExt;
use futures::stream::FuturesUnordered;

pub(crate) async fn centralized_all_reduce_sum<B, C, S>(
    data_service: &TensorDataClient<B, C, S>,
    tensor: B::FloatTensorPrimitive,
    device: &B::Device,
    strategy: CentralizedAllReduceStrategy,
) -> B::FloatTensorPrimitive
where
    B: Backend,
    C: NetworkClient,
    S: NetworkServer<State = Arc<TensorDataService<B, C>>>,
{
    match strategy {
        Central { other_nodes } => {
            // Transfer 1: download tensors from other nodes
            let mut futures = other_nodes
                .iter()
                .map(|x| {
                    let device = device.clone(); // if device is Clone, otherwise ref
                    let data_service = data_service.clone();
                    async move {
                        let data = data_service
                            .download_tensor(x, 0.into())
                            .await
                            .expect("Couldn't find the tensor for transfer id 0");
                        B::float_from_data(data, &device)
                    }
                })
                .collect::<FuturesUnordered<_>>();

            // Sum all downloads async
            let mut sum = tensor;
            while let Some(res) = futures.next().await {
                // If the tensor is empty, we can skip it
                sum = B::float_add(sum, res);
            }

            // Transfer 2: Expose result
            data_service
                .expose(sum.clone(), other_nodes.len() as u32, 1.into())
                .await;

            sum
        }
        Peripheral { central_node } => {
            // Transfer 1: Expose input
            data_service.expose(tensor, 1, 0.into()).await;

            // Transfer 2: Download result
            let data = data_service
                .download_tensor(&central_node, 1.into())
                .await
                .expect("Couldn't find the tensor for transfer id 1");

            B::float_from_data(data, device)
        }
    }
}
