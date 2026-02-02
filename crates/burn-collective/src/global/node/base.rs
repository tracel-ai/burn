use burn_communication::Protocol;
use burn_communication::data_service::TensorDataServer;
use burn_communication::{Address, ProtocolServer, data_service::TensorDataService};
use burn_tensor::backend::Backend;
use std::collections::HashMap;
use std::{marker::PhantomData, sync::Arc};
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

use crate::node::sync::SyncService;
use crate::{
    AllReduceStrategy, BroadcastStrategy, GlobalRegisterParams, NodeId, PeerId, ReduceStrategy,
};
use crate::{
    ReduceOperation,
    global::{
        node::{
            centralized::centralized_all_reduce_sum, ring::ring_all_reduce_sum,
            tree::tree_all_reduce_sum, worker::GlobalClientWorker,
        },
        shared::{GlobalCollectiveError, RemoteRequest, RemoteResponse},
    },
    local::server::get_collective_server_runtime,
};

/// Must be synchronized between all nodes for collective operations to work
pub(crate) struct NodeState {
    pub node_id: NodeId,
    pub nodes: HashMap<NodeId, Address>,
    pub num_global_devices: u32,
}

/// A node talks to the global orchestrator as well as other nodes with a peer-to-peer service
pub(crate) struct Node<B, P>
where
    B: Backend,
    P: Protocol,
{
    // State is written during `register` and read during other operations,
    // sometimes by multiple threads (ex. syncing during an all-reduce)
    state: Arc<RwLock<Option<NodeState>>>,
    data_service: Arc<TensorDataService<B, P>>,
    sync_service: Arc<SyncService<P>>,
    worker: GlobalClientWorker<P::Client>,
    _n: PhantomData<P>,
}

impl<B, P> Node<B, P>
where
    B: Backend,
    P: Protocol,
{
    pub fn new(global_address: &Address, comms_server: P::Server) -> Self {
        let state = Arc::new(tokio::sync::RwLock::new(None));
        let cancel_token = CancellationToken::new();
        let data_service = Arc::new(TensorDataService::new(cancel_token.clone()));
        let sync_service = Arc::new(SyncService::new(state.clone()));

        let runtime = get_collective_server_runtime();
        let server = comms_server
            .route_tensor_data_service(data_service.clone())
            .route("/sync", {
                let sync_service = sync_service.clone();
                async move |channel: <P::Server as ProtocolServer>::Channel| {
                    sync_service.handle_sync_connection(channel).await;
                }
            })
            .serve({
                let cancel_token = cancel_token.clone();
                async move { cancel_token.cancelled().await }
            });

        runtime.spawn(server);

        let worker = GlobalClientWorker::new(&runtime, cancel_token.clone(), global_address);

        Self {
            state,
            data_service,
            sync_service,
            worker,
            _n: PhantomData,
        }
    }

    pub async fn register(
        &mut self,
        peers: Vec<PeerId>,
        global_params: GlobalRegisterParams,
    ) -> Result<(), GlobalCollectiveError> {
        let req = RemoteRequest::Register {
            node_addr: global_params.node_address,
            num_nodes: global_params.num_nodes,
            peers,
        };
        match self.worker.request(req).await {
            RemoteResponse::Register {
                node_id,
                nodes,
                num_global_devices,
            } => {
                let mut state = self.state.write().await;
                *state = Some(NodeState {
                    node_id,
                    nodes,
                    num_global_devices,
                });
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

    /// Performs an all-reduce
    ///
    /// Reads the NodeState
    pub async fn all_reduce(
        &self,
        tensor: B::FloatTensorPrimitive,
        strategy: AllReduceStrategy,
        op: ReduceOperation,
    ) -> Result<B::FloatTensorPrimitive, GlobalCollectiveError> {
        let state = self.state.read().await;
        let Some(ref state) = *state else {
            return Err(GlobalCollectiveError::AllReduceBeforeRegister);
        };
        let node = state.node_id;
        let nodes = &state.nodes;

        let mut result = match strategy {
            AllReduceStrategy::Centralized => {
                centralized_all_reduce_sum(
                    node,
                    nodes,
                    &self.data_service,
                    self.sync_service.clone(),
                    tensor,
                )
                .await?
            }
            AllReduceStrategy::Tree(arity) => {
                tree_all_reduce_sum(
                    node,
                    nodes,
                    self.data_service.clone(),
                    self.sync_service.clone(),
                    tensor,
                    arity,
                )
                .await?
            }
            AllReduceStrategy::Ring => {
                ring_all_reduce_sum(
                    node,
                    nodes,
                    self.data_service.clone(),
                    self.sync_service.clone(),
                    tensor,
                )
                .await?
            }
        };

        if op == ReduceOperation::Mean {
            result = B::float_div_scalar(result, (state.num_global_devices as f32).into());
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
        unimplemented!("Global reduce unimplemented");
    }

    pub async fn broadcast(
        &self,
        _tensor: Option<B::FloatTensorPrimitive>,
        _strategy: BroadcastStrategy,
    ) -> Result<B::FloatTensorPrimitive, GlobalCollectiveError> {
        unimplemented!("Global broadcast unimplemented");
    }

    pub async fn finish(&mut self) {
        let res = self.worker.close_connection().await;
        if let Err(err) = res {
            log::error!("Global collective client error: {err:?}");
        }
        self.data_service.close().await;
    }
}
