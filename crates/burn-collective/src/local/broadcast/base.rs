use std::{collections::HashMap, sync::mpsc::SyncSender};

use burn_communication::websocket::WebSocket;
use burn_tensor::backend::Backend;

use crate::{
    BroadcastStrategy, CollectiveConfig, CollectiveError, PeerId,
    local::{broadcast_centralized, broadcast_tree},
    node::base::Node,
};

/// An on-going broadcast operation
pub struct BroadcastOp<B: Backend> {
    /// broadcast calls, one for each calling device
    calls: Vec<BroadcastOpCall<B>>,
    /// The tensor to broadcast, as defined by the root. Should be defined before all
    /// peers call the operation.
    tensor: Option<B::FloatTensorPrimitive>,
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
    pub async fn execute(
        mut self,
        config: &CollectiveConfig,
        global_client: &mut Option<Node<B, WebSocket>>,
    ) {
        // all registered callers have sent a tensor to aggregate
        let tensors = self.broadcast(config, global_client).await;
        match tensors {
            Ok(mut tensors) => {
                // Return resulting tensors
                self.calls.drain(..).for_each(|op| {
                    let result = tensors.remove(&op.caller).unwrap();
                    op.result_sender.send(Ok(result)).unwrap();
                });
            }
            Err(err) => {
                // Send error to all subscribers
                self.send_err_to_all(err);
            }
        }
    }

    async fn broadcast(
        &mut self,
        config: &CollectiveConfig,
        global_client: &mut Option<Node<B, WebSocket>>,
    ) -> Result<HashMap<PeerId, B::FloatTensorPrimitive>, CollectiveError> {
        let local_strategy = config.local_broadcast_strategy;

        // Get corresponding devices for each peer
        let devices = self
            .calls
            .iter()
            .map(|op| (op.caller, op.device.clone()))
            .collect::<HashMap<PeerId, B::Device>>();

        // Chose a root
        let root = self.root.unwrap_or(self.calls.first().unwrap().caller);

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
        let results = match local_strategy {
            BroadcastStrategy::Tree(arity) => broadcast_tree::<B>(devices, root, tensor, arity),
            BroadcastStrategy::Centralized => broadcast_centralized::<B>(devices, root, tensor),
        };

        Ok(results)
    }

    /// Send a collective error as result to operation caller
    pub fn send_err_to_all(&mut self, err: CollectiveError) {
        self.calls.drain(..).for_each(|op| {
            op.result_sender.send(Err(err.clone())).unwrap();
        });
    }
}
