use std::sync::Arc;

use crate::global::server::base::GlobalCollectiveError;
use crate::global::shared::base::CentralizedAllReduceStrategy;
use crate::global::shared::base::CentralizedAllReduceStrategy::{Central, Peripheral};
use burn_communication::data_service::TensorDataService;
use burn_communication::network::Network;
use burn_tensor::TensorMetadata;
use burn_tensor::backend::Backend;
use futures::StreamExt;
use futures::stream::FuturesUnordered;

pub(crate) async fn centralized_all_reduce_sum<B, N>(
    data_service: &Arc<TensorDataService<B, N>>,
    tensor: B::FloatTensorPrimitive,
    device: &B::Device,
    strategy: CentralizedAllReduceStrategy,
) -> Result<B::FloatTensorPrimitive, GlobalCollectiveError>
where
    B: Backend,
    N: Network,
{
    let shape = tensor.shape();

    match strategy {
        Central { other_nodes } => {
            // Transfer 1: download tensors from other nodes
            let mut futures = other_nodes
                .iter()
                .map(|address| {
                    let device = device.clone(); // if device is Clone, otherwise ref
                    let data_service = data_service.clone();
                    async move {
                        let data = data_service
                            .download_tensor(address.clone(), 0.into())
                            .await
                            .expect("Couldn't find the tensor for transfer id 0");
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
            data_service
                .expose(sum.clone(), other_nodes.len() as u32, 1.into())
                .await;

            Ok(sum)
        }
        Peripheral { central_node } => {
            // Transfer 1: Expose input
            data_service.expose(tensor, 1, 0.into()).await;

            // Transfer 2: Download result
            let data = data_service
                .download_tensor(central_node, 1.into())
                .await
                .expect("Couldn't find the tensor for transfer id 1");

            let res = B::float_from_data(data, device);
            if shape != res.shape() {
                return Err(GlobalCollectiveError::PeerSentIncoherentTensor);
            }
            Ok(res)
        }
    }
}
