use burn_network::network::{NetworkClient, NetworkServer};
use burn_tensor::{ElementConversion, backend::Backend};
use std::{marker::PhantomData, sync::Arc};
use tokio_util::sync::CancellationToken;

use crate::{
    GlobalAllReduceParams, ReduceKind, RegisterParams,
    global::{
        client::{
            centralized::centralized_all_reduce_sum,
            data_server::{TensorDataClient, TensorDataService},
            ring::ring_all_reduce_sum,
            tree::tree_all_reduce_sum,
            worker::GlobalClientWorker,
        },
        shared::base::{NodeAddress, RemoteRequest, RemoteResponse},
    },
    local_server::get_server_runtime,
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
    num_global_devices: Option<u32>,
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

        let runtime = get_server_runtime();

        let data_client = TensorDataClient::new(&runtime, cancel_token.clone(), data_server_port);

        let worker = GlobalClientWorker::new(&runtime, cancel_token.clone(), server_address);

        Self {
            data_service: data_client,
            data_client_address: Arc::new(NodeAddress(client_address.to_owned())),
            worker,
            num_global_devices: None,
            _phantom_data: PhantomData,
        }
    }

    pub async fn register(&mut self, params: RegisterParams) {
        let node_addr = self.data_client_address.as_ref().clone();
        let global_params = params
            .global_params
            .expect("Must have global params for global register");
        let req = RemoteRequest::Register {
            node_id: global_params.node_id,
            node_addr,
            num_nodes: global_params.num_nodes,
            num_local_devices: params.num_devices,
        };
        let resp = self.worker.request(req).await;
        let RemoteResponse::RegisterAck { num_global_devices } = resp else {
            panic!("The response to a register request should be a RegisterAck, not {resp:?}");
        };
        self.num_global_devices = Some(num_global_devices);
    }

    pub async fn all_reduce(
        &self,
        tensor: B::FloatTensorPrimitive,
        params: GlobalAllReduceParams,
        device: &B::Device,
        kind: ReduceKind,
    ) -> B::FloatTensorPrimitive {
        let num_global_devices = self
            .num_global_devices
            .expect("Can't all-reduce before registering (global)");

        // Get strategy from server
        let req = RemoteRequest::AllReduce {
            params: params.clone(),
        };
        let resp = self.worker.request(req).await;

        let mut result = match resp {
            RemoteResponse::CentralizedAllReduceStrategy(strategy) => {
                centralized_all_reduce_sum(&self.data_service, tensor, device, strategy).await
            }
            RemoteResponse::TreeAllReduceStrategy(strategy) => {
                tree_all_reduce_sum(&self.data_service, tensor, device, strategy).await
            }
            RemoteResponse::RingAllReduceStrategy(strategy) => {
                ring_all_reduce_sum(&self.data_service, tensor, device, strategy).await
            }
            RemoteResponse::Error(err) => panic!("Global collective server error: {err}"),
            resp => {
                panic!("The response to a all-reduce request should be a strategy, not {resp:?}")
            }
        };

        if kind == ReduceKind::Mean {
            result = B::float_div_scalar(result, (num_global_devices).elem());
        }

        result
    }

    pub async fn finish(&mut self) {
        self.worker.close_connection().await;
        self.data_service.close().await;
    }
}
