use crate::local::tensor_map::{CollectiveTensorMap, PeerDeviceMap};
use crate::{
    BroadcastStrategy, CollectiveConfig, CollectiveError, PeerId,
    local::{broadcast_centralized, broadcast_tree},
    node::base::Node,
};
use burn_communication::Protocol;
#[allow(unused_imports)]
use burn_tensor::TensorMetadata;
use burn_tensor::backend::Backend;
use std::sync::mpsc::SyncSender;

/// An on-going broadcast operation
pub struct BroadcastOp<B: Backend> {
    /// broadcast calls, one for each calling device
    calls: Vec<BroadcastOpCall<B>>,
    /// The tensor to broadcast, as defined by the root. Should be defined before all
    /// peers call the operation.
    tensor: Option<B::FloatTensorPrimitive>,

    /// ID of the root (or use the first call's peer).
    root: Option<PeerId>,
}

/// Struct for each device that calls an broadcast operation
pub struct BroadcastOpCall<B: Backend> {
    /// Id of the caller of the operation
    caller: PeerId,
    /// Device of the calling peer
    device: B::Device,
    /// Callback for the result of the broadcast
    result_sender: SyncSender<BroadcastResult<B::FloatTensorPrimitive>>,
}

/// Type sent to the collective client upon completion of a broadcast op
pub(crate) type BroadcastResult<T> = Result<T, CollectiveError>;

impl<B: Backend> BroadcastOp<B> {
    pub fn new() -> Self {
        Self {
            calls: vec![],
            tensor: None,
            root: None,
        }
    }

    /// Get the effective root of the broadcast operation.
    /// If the root is set, return it. Otherwise, return the first caller's peer.
    pub fn effective_root(&self) -> PeerId {
        self.root.unwrap_or(self.calls.first().unwrap().caller)
    }

    pub fn peers(&self) -> Vec<PeerId> {
        self.calls.iter().map(|c| c.caller).collect()
    }

    fn peer_devices(&self) -> PeerDeviceMap<B> {
        self.calls
            .iter()
            .map(|call| (call.caller, call.device.clone()))
            .collect()
    }

    /// Register a call to reduce in this operation.
    /// When the last caller registers a reduce, the operation is executed.
    pub fn register_call(
        &mut self,
        caller: PeerId,
        input: Option<B::FloatTensorPrimitive>,
        result_sender: SyncSender<BroadcastResult<B::FloatTensorPrimitive>>,
        device: B::Device,
        peer_count: usize,
    ) -> Result<bool, CollectiveError> {
        if input.is_some() {
            if self.tensor.is_some() {
                return Err(CollectiveError::BroadcastMultipleTensors);
            }
            self.tensor = input;
        }

        self.calls.push(BroadcastOpCall {
            caller,
            device,
            result_sender,
        });

        Ok(self.calls.len() == peer_count)
    }

    /// Runs the broadcast if the operation is ready. Otherwise, do nothing
    #[tracing::instrument(
        skip(self, config, global_client),
        fields(
            self.peers = ?self.peers(),
            self.shape = ?self.tensor.as_ref().map(|t| t.shape()),
            self.dtype = ?self.tensor.as_ref().map(|t| t.dtype()),
        )
    )]
    pub async fn execute<P: Protocol>(
        mut self,
        config: &CollectiveConfig,
        global_client: &mut Option<Node<B, P>>,
    ) {
        // all registered callers have sent a tensor to aggregate
        match self.broadcast(config, global_client).await {
            Ok(mut tensors) => {
                // Return resulting tensors
                self.calls.iter().for_each(|call| {
                    let result = tensors
                        .remove(&call.caller)
                        .expect("tensor/peer internal mismatch.");
                    call.result_sender.send(Ok(result)).unwrap();
                });
                assert_eq!(tensors.len(), 0, "tensor/peer internal mismatch.");
            }
            Err(err) => {
                // Send error to all subscribers
                self.fail(err);
            }
        }
    }

    #[tracing::instrument(skip(self, config, global_client))]
    async fn broadcast<P: Protocol>(
        &mut self,
        config: &CollectiveConfig,
        global_client: &mut Option<Node<B, P>>,
    ) -> Result<CollectiveTensorMap<B>, CollectiveError> {
        let local_strategy = config.local_broadcast_strategy;

        let peer_devices = self.peer_devices();

        let root = self.effective_root();

        // Do broadcast on global level with the main tensor
        if let Some(global_client) = &global_client {
            let global_strategy = config.global_broadcast_strategy.unwrap();
            let global_result = global_client
                .broadcast(self.tensor.clone(), global_strategy)
                .await
                .map_err(CollectiveError::Global)?;
            self.tensor = Some(global_result)
        }

        // At this point tensor must be defined
        let Some(tensor) = self.tensor.take() else {
            return Err(CollectiveError::BroadcastNoTensor);
        };

        // Broadcast locally
        Ok(match local_strategy {
            BroadcastStrategy::Tree(arity) => {
                broadcast_tree::<B>(peer_devices, root, tensor, arity)
            }
            BroadcastStrategy::Centralized => {
                broadcast_centralized::<B>(peer_devices, root, tensor)
            }
        })
    }

    /// Send a collective error as result to operation caller
    pub fn fail(self, err: CollectiveError) {
        self.calls.iter().for_each(|call| {
            call.result_sender.send(Err(err.clone())).unwrap();
        });
    }
}
