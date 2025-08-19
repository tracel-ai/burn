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

impl<B: Backend> BroadcastOpCall<B> {
    pub fn new(
        caller: PeerId,
        device: B::Device,
        result_sender: SyncSender<BroadcastResult<B::FloatTensorPrimitive>>,
    ) -> Self {
        Self {
            caller,
            device,
            result_sender,
        }
    }
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

    /// Register a call to broadcast in this operation.
    /// When the last caller registers an broadcast, the operation is executed.
    pub async fn register_call(
        &mut self,
        call: BroadcastOpCall<B>,
        tensor: Option<B::FloatTensorPrimitive>,
        peers: &[PeerId],
        config: &CollectiveConfig,
        global_client: &mut Option<Node<B, WebSocket>>,
    ) {
        if tensor.is_some() {
            if self.tensor.is_some() {
                self.send_err_to_all(CollectiveError::BroadcastMultipleTensors);
            }
            self.tensor = tensor;
        }

        self.calls.push(call);

        let tensor_count = self.calls.len();
        if tensor_count == peers.len() {
            // Do broadcast
            match self.broadcast(config, global_client).await {
                Ok(mut result) => {
                    // Return resulting tensors to each corresponding peer
                    self.calls.drain(..).for_each(|op| {
                        op.result_sender
                            .send(Ok(result.remove(&op.caller).unwrap()))
                            .unwrap();
                    });
                }
                Err(err) => {
                    self.send_err_to_all(err);
                }
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
    fn send_err_to_all(&mut self, err: CollectiveError) {
        self.calls.drain(..).for_each(|op| {
            op.result_sender.send(Err(err.clone())).unwrap();
        });
    }
}
