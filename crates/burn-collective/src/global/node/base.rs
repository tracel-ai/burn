use burn_communication::Protocol;
use burn_communication::data_service::TensorDataServer;
use burn_communication::{Address, ProtocolServer, data_service::TensorDataService};
use burn_tensor::{ElementConversion, backend::Backend};
use std::{marker::PhantomData, sync::Arc};
use tokio_util::sync::CancellationToken;

use crate::{AllReduceStrategy, BroadcastStrategy, GlobalRegisterParams, PeerId, ReduceStrategy};
use crate::{
    ReduceOperation,
    global::{
        node::{
            centralized::centralized_all_reduce_sum, ring::ring_all_reduce_sum,
            tree::tree_all_reduce_sum, worker::GlobalClientWorker,
        },
        shared::{GlobalCollectiveError, RemoteRequest, RemoteResponse},
    },
    local_server::get_server_runtime,
};

/// The client that talks to the global collective orchestrator
pub(crate) struct GlobalCollectiveClient<B, N>
where
    B: Backend,
    N: Protocol,
{
    data_service: Arc<TensorDataService<B, N>>,
    node_address: Arc<Address>,
    worker: GlobalClientWorker<N::Client>,
    num_global_devices: Option<u32>,
    _n: PhantomData<N>,
}

impl<B, N> GlobalCollectiveClient<B, N>
where
    B: Backend,
    N: Protocol,
{
    pub fn new(global_address: &Address, node_address: &Address, comms_server: N::Server) -> Self {
        let cancel_token = CancellationToken::new();

        let data_service = Arc::new(TensorDataService::new(cancel_token.clone()));

        let runtime = get_server_runtime();
        let server = comms_server
            .route_tensor_data_service(data_service.clone())
            .serve({
                let cancel_token = cancel_token.clone();
                async move { cancel_token.cancelled().await }
            });

        runtime.spawn(server);

        let worker = GlobalClientWorker::new(&runtime, cancel_token.clone(), global_address);

        Self {
            data_service,
            node_address: Arc::new(node_address.clone()),
            worker,
            num_global_devices: None,
            _n: PhantomData,
        }
    }

    pub async fn register(
        &mut self,
        peers: Vec<PeerId>,
        global_params: GlobalRegisterParams,
    ) -> Result<(), GlobalCollectiveError> {
        let node_addr = self.node_address.as_ref().clone();

        let req = RemoteRequest::Register {
            node_id: global_params.node_id,
            node_addr,
            num_nodes: global_params.num_nodes,
            peers,
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
                return Err(GlobalCollectiveError::WrongOrchestratorResponse);
            }
        }

        Ok(())
    }

    pub async fn all_reduce(
        &self,
        tensor: B::FloatTensorPrimitive,
        strategy: AllReduceStrategy,
        device: &B::Device, //TODO remove this!
        op: ReduceOperation,
    ) -> Result<B::FloatTensorPrimitive, GlobalCollectiveError> {
        let num_global_devices = self
            .num_global_devices
            .expect("Can't all-reduce before registering (global)");

        // Get strategy from orchestrator
        let req = RemoteRequest::AllReduce { strategy };
        let resp = self.worker.request(req).await;

        let mut result = match resp {
            RemoteResponse::CentralizedAllReduceStrategy(strategy) => {
                centralized_all_reduce_sum(&self.data_service, tensor, device, strategy).await?
            }
            RemoteResponse::TreeAllReduceStrategy(strategy) => {
                tree_all_reduce_sum(self.data_service.clone(), tensor, device, strategy).await?
            }
            RemoteResponse::RingAllReduceStrategy(strategy) => {
                ring_all_reduce_sum(self.data_service.clone(), tensor, device, strategy).await?
            }
            RemoteResponse::Error(err) => {
                return Err(err);
            }
            resp => {
                log::error!("Response to a all-reduce request should be a strategy, not {resp:?}");
                return Err(GlobalCollectiveError::WrongOrchestratorResponse);
            }
        };

        if op == ReduceOperation::Mean {
            result = B::float_div_scalar(result, (num_global_devices).elem());
        }

        Ok(result)
    }

    pub async fn reduce(
        &self,
        _tensor: B::FloatTensorPrimitive,
        _strategy: ReduceStrategy,
        _root: PeerId,
        _op: ReduceOperation,
    ) -> Result<Option<B::FloatTensorPrimitive>, GlobalCollectiveError> {
        unimplemented!("TODO");
    }

    pub async fn broadcast(
        &self,
        _tensor: Option<B::FloatTensorPrimitive>,
        _strategy: BroadcastStrategy,
        _root: PeerId,
    ) -> Result<B::FloatTensorPrimitive, GlobalCollectiveError> {
        unimplemented!("TODO");
    }

    pub async fn finish(&mut self) {
        let res = self.worker.close_connection().await;
        if let Err(err) = res {
            log::error!("Global collective client error: {err:?}");
        }
        self.data_service.close().await;
    }
}
