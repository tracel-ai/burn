use burn_network::data_service::TensorDataServer;
use burn_network::network::Network;
use burn_network::{
    data_service::TensorDataService,
    network::{NetworkAddress, NetworkServer},
};
use burn_tensor::{ElementConversion, backend::Backend};
use std::{marker::PhantomData, sync::Arc};
use tokio_util::sync::CancellationToken;

use crate::global::server::base::GlobalCollectiveError;
use crate::{
    GlobalAllReduceParams, ReduceKind, RegisterParams,
    global::{
        client::{
            centralized::centralized_all_reduce_sum, ring::ring_all_reduce_sum,
            tree::tree_all_reduce_sum, worker::GlobalClientWorker,
        },
        shared::base::{RemoteRequest, RemoteResponse},
    },
    local_server::get_server_runtime,
};

pub(crate) struct GlobalCollectiveClient<B, N>
where
    B: Backend,
    N: Network,
{
    data_service: Arc<TensorDataService<B, N>>,
    data_client_address: Arc<NetworkAddress>,
    worker: GlobalClientWorker<N::Client>,
    num_global_devices: Option<u32>,
    _n: PhantomData<N>,
}

impl<B, N> GlobalCollectiveClient<B, N>
where
    B: Backend,
    N: Network,
{
    pub fn new(
        server_address: &NetworkAddress,
        client_address: &NetworkAddress,
        data_server_port: u16,
    ) -> Self {
        let cancel_token = CancellationToken::new();

        let data_service = Arc::new(TensorDataService::new(cancel_token.clone()));

        let runtime = get_server_runtime();
        let server = N::Server::new(data_server_port)
            .route_tensor_data_service(data_service.clone())
            .serve({
                let cancel_token = cancel_token.clone();
                async move { cancel_token.cancelled().await }
            });

        runtime.spawn(server);

        let worker = GlobalClientWorker::new(&runtime, cancel_token.clone(), server_address);

        Self {
            data_service,
            data_client_address: Arc::new(client_address.clone()),
            worker,
            num_global_devices: None,
            _n: PhantomData,
        }
    }

    pub async fn register(&mut self, params: RegisterParams) -> Result<(), GlobalCollectiveError> {
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
        match self.worker.request(req).await {
            RemoteResponse::RegisterAck { num_global_devices } => {
                self.num_global_devices = Some(num_global_devices);
            }
            RemoteResponse::Error(err) => {
                return Err(err);
            }
            resp => {
                log::error!("Response to a register request should be an ack, not {resp:?}");
                return Err(GlobalCollectiveError::WrongServerResponse);
            }
        }

        Ok(())
    }

    pub async fn all_reduce(
        &self,
        tensor: B::FloatTensorPrimitive,
        params: GlobalAllReduceParams,
        device: &B::Device,
        kind: ReduceKind,
    ) -> Result<B::FloatTensorPrimitive, GlobalCollectiveError> {
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
                tree_all_reduce_sum(self.data_service.clone(), tensor, device, strategy).await
            }
            RemoteResponse::RingAllReduceStrategy(strategy) => {
                ring_all_reduce_sum(self.data_service.clone(), tensor, device, strategy).await
            }
            RemoteResponse::Error(err) => {
                return Err(err);
            }
            resp => {
                log::error!("Response to a all-reduce request should be a strategy, not {resp:?}");
                return Err(GlobalCollectiveError::WrongServerResponse);
            }
        };

        if kind == ReduceKind::Mean {
            result = B::float_div_scalar(result, (num_global_devices).elem());
        }

        Ok(result)
    }

    pub async fn finish(&mut self) {
        let res = self.worker.close_connection().await;
        if let Err(err) = res {
            log::error!("Global collective client error: {err:?}");
        }
        self.data_service.close().await;
    }
}
