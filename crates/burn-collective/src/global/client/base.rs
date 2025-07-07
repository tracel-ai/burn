use burn_network::network::{NetworkClient, NetworkServer};
use burn_tensor::backend::Backend;
use std::{marker::PhantomData, sync::Arc};
use tokio::runtime::Runtime;
use tokio_util::sync::CancellationToken;

use crate::{
    GlobalAllReduceParams, GlobalRegisterParams,
    global::{
        client::{
            centralized::centralized_all_reduce,
            data_server::{TensorDataClient, TensorDataService},
            ring::ring_all_reduce,
            tree::tree_all_reduce,
            worker::GlobalClientWorker,
        },
        shared::base::{NodeAddress, RemoteRequest, RemoteResponse},
    },
};

pub(crate) struct GlobalCollectiveClient<B, C, S>
where
    B: Backend,
    C: NetworkClient,
    S: NetworkServer<State = Arc<TensorDataService<B, C>>>,
{
    data_service: TensorDataClient<B, C, S>,
    data_client_address: Arc<NodeAddress>,
    worker: GlobalClientWorker<C>,
    _runtime: Runtime,
    _phantom_data: PhantomData<B>,
}

impl<B, C, S> GlobalCollectiveClient<B, C, S>
where
    B: Backend,
    C: NetworkClient,
    S: NetworkServer<State = Arc<TensorDataService<B, C>>>,
{
    pub fn new(server_address: &str, client_address: &str, data_server_port: u16) -> Self {
        let cancel_token = CancellationToken::new();

        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();

        let data_client = TensorDataClient::new(&runtime, cancel_token.clone(), data_server_port);

        let worker = GlobalClientWorker::new(&runtime, cancel_token.clone(), server_address);

        Self {
            data_service: data_client,
            data_client_address: Arc::new(NodeAddress(client_address.to_owned())),
            worker,
            _runtime: runtime,
            _phantom_data: PhantomData,
        }
    }

    pub async fn register(&mut self, node_id: u32, params: GlobalRegisterParams) {
        let node_addr = self.data_client_address.as_ref().clone();
        let req = RemoteRequest::Register {
            node_id,
            node_addr,
            num_nodes: params.num_nodes,
        };
        let resp = self.worker.request(req).await;
        if resp != RemoteResponse::RegisterAck {
            panic!(
                "The response to a register request should be a RegisterAck, not {:?}",
                resp
            );
        }
    }

    pub async fn all_reduce(
        &mut self,
        tensor: B::FloatTensorPrimitive,
        params: GlobalAllReduceParams,
        device: &B::Device,
    ) -> B::FloatTensorPrimitive {
        let req = RemoteRequest::AllReduce { params };
        let resp = self.worker.request(req).await;
        match resp {
            RemoteResponse::CentralizedAllReduceStrategy(strategy) => {
                return centralized_all_reduce(&self.data_service, tensor, device, strategy).await;
            }
            RemoteResponse::TreeAllReduceStrategy(strategy) => {
                return tree_all_reduce(&self.data_service, tensor, device, strategy).await;
            }
            RemoteResponse::RingAllReduceStrategy(strategy) => {
                return ring_all_reduce(&self.data_service, tensor, device, strategy).await;
            }
            RemoteResponse::Error(err) => panic!("Global collective server error: {err}"),
            resp => panic!(
                "The response to a register request should be a RegisterAck, not {:?}",
                resp
            ),
        };
    }

    pub async fn finish(&mut self) {
        self.worker.close_connection().await;
        self.data_service.close().await;
    }
}
