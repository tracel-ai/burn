use std::sync::Arc;

use crate::global::{
    client::data_server::{TensorDataClient, TensorDataService},
    shared::base::TreeAllReduceStrategy,
};
use burn_network::network::{NetworkClient, NetworkServer};
use burn_tensor::backend::Backend;
use futures::{StreamExt, stream::FuturesUnordered};

pub(crate) async fn tree_all_reduce_sum<B, C, S>(
    data_service: &TensorDataClient<B, C, S>,
    tensor: B::FloatTensorPrimitive,
    device: &B::Device,
    strategy: TreeAllReduceStrategy,
) -> B::FloatTensorPrimitive
where
    B: Backend,
    C: NetworkClient,
    S: NetworkServer<State = Arc<TensorDataService<B, C>>>,
{
    // Transfer #1: Download tensors from children async
    let mut downloads = strategy
        .children
        .iter()
        .map(|child| {
            let data_service = data_service.clone();
            async move {
                let data = data_service.download_tensor(child, 0.into()).await.unwrap();

                B::float_from_data(data, device)
            }
        })
        .collect::<FuturesUnordered<_>>();

    // Sum download results
    let mut result = tensor;
    while let Some(res) = downloads.next().await {
        result = B::float_add(result, res);
    }

    // Transfer #1: Expose the result to the parent
    if let Some(parent) = &strategy.parent {
        data_service.expose(result.clone(), 1, 0.into()).await;

        // Transfer #2: Download final tensor from parent
        let data = data_service
            .download_tensor(parent, 1.into())
            .await
            .unwrap();
        let parent_tensor = B::float_from_data(data, device);
        result = parent_tensor;
    }

    // Tranfer #2: Expose the final result to all children
    if !strategy.children.is_empty() {
        data_service
            .expose(result.clone(), strategy.children.len() as u32 + 1, 1.into())
            .await;
    }

    result
}
