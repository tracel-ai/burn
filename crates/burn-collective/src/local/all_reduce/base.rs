use std::{collections::HashMap, sync::mpsc::SyncSender};

use burn_communication::websocket::WebSocket;
use burn_tensor::{ElementConversion, Shape, TensorMetadata, backend::Backend};

use crate::{
    AllReduceStrategy, CollectiveConfig, CollectiveError, PeerId, ReduceOperation,
    local::{
        all_reduce_sum_centralized, all_reduce_sum_ring, all_reduce_sum_tree,
        broadcast_centralized, broadcast_tree, reduce_sum_centralized, reduce_sum_tree,
    },
    node::base::Node,
};

/// An on-going all-reduce operation
pub struct AllReduceOp<B: Backend> {
    /// all-reduce calls, one for each calling device
    calls: Vec<AllReduceOpCall<B>>,
    /// The reduce operation of the current all-reduce, as defined by the first caller
    op: ReduceOperation,
    /// The shape of the current all-reduce, as defined by the first caller
    shape: Shape,
}

/// Struct for each device that calls an all-reduce operation
pub struct AllReduceOpCall<B: Backend> {
    /// Id of the caller for this operation
    caller: PeerId,
    /// The tensor primitive passed as input
    input: B::FloatTensorPrimitive,
    /// Callback for the result of the all-reduce
    result_sender: SyncSender<AllReduceResult<B::FloatTensorPrimitive>>,
}

/// Type sent to the collective client upon completion of a all-reduce aggregation
pub(crate) type AllReduceResult<T> = Result<T, CollectiveError>;

impl<B: Backend> AllReduceOp<B> {
    pub fn new(shape: Shape, reduce_op: ReduceOperation) -> Self {
        Self {
            calls: vec![],
            op: reduce_op,
            shape,
        }
    }

    /// Register a call to all-reduce in this operation.
    ///
    /// # Returns
    ///
    /// `true` if enough peers have registered, and the all-reduce is ready
    pub fn register_call(
        &mut self,
        caller: PeerId,
        input: B::FloatTensorPrimitive,
        result_sender: SyncSender<AllReduceResult<B::FloatTensorPrimitive>>,
        op: ReduceOperation,
        peer_count: usize,
    ) -> Result<bool, CollectiveError> {
        if self.shape != input.shape() {
            return Err(CollectiveError::AllReduceShapeMismatch);
        }
        if self.op != op {
            return Err(CollectiveError::AllReduceOperationMismatch);
        }

        self.calls.push(AllReduceOpCall {
            caller,
            input,
            result_sender,
        });

        Ok(self.calls.len() == peer_count)
    }

    /// Runs the all-reduce if the operation is ready. Otherwise, do nothing
    pub async fn execute(
        mut self,
        config: &CollectiveConfig,
        global_client: &mut Option<Node<B, WebSocket>>,
    ) {
        // all registered callers have sent a tensor to aggregate
        let tensors = self.all_reduce(config, global_client).await;
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

    /// Perform an all-reduce operation.
    async fn all_reduce(
        &mut self,
        config: &CollectiveConfig,
        global_client: &mut Option<Node<B, WebSocket>>,
    ) -> Result<HashMap<PeerId, B::FloatTensorPrimitive>, CollectiveError> {
        let mut tensors = HashMap::new();
        for call in &mut self.calls {
            tensors.insert(call.caller, call.input.clone());
        }

        let op = self.op;
        if let Some(global_client) = global_client.as_mut() {
            Self::all_reduce_with_global(&mut tensors, op, config, global_client).await?;
        } else {
            Self::all_reduce_local_only(&mut tensors, op, config).await?;
        }

        Ok(tensors)
    }

    /// Perform an all-reduce with no multi-node operations (global ops)
    async fn all_reduce_local_only(
        tensors: &mut HashMap<PeerId, B::FloatTensorPrimitive>,
        op: ReduceOperation,
        config: &CollectiveConfig,
    ) -> Result<(), CollectiveError> {
        let local_strategy = &config.local_all_reduce_strategy;
        match local_strategy {
            AllReduceStrategy::Centralized => all_reduce_sum_centralized::<B>(tensors),
            AllReduceStrategy::Tree(arity) => all_reduce_sum_tree::<B>(tensors, *arity),
            AllReduceStrategy::Ring => all_reduce_sum_ring::<B>(tensors),
        };

        if op == ReduceOperation::Mean {
            // Apply mean division
            let tensor_count = tensors.len() as f32;
            tensors.iter_mut().for_each(|(_, tensor)| {
                *tensor = B::float_div_scalar(tensor.clone(), tensor_count.elem())
            });
        }

        Ok(())
    }

    /// Do an all-reduce in a multi-node context
    ///
    /// With Tree and Centralized strategies, the all-reduce is split between a
    /// reduce (all tensors are reduced to one device), and a broadcast (the result is sent to all
    /// other devices). The all-reduce on the global level is done between both steps.
    /// Due to the nature of the Ring strategy, this separation can't be done.
    // For the Ring strategy, this isn't possible, because it is more like a
    // reduce-scatter plus an all-gather, so using a Ring strategy locally in a multi-node
    // setup may be unadvantageous.
    async fn all_reduce_with_global(
        tensors: &mut HashMap<PeerId, B::FloatTensorPrimitive>,
        op: ReduceOperation,
        config: &CollectiveConfig,
        global_client: &mut Node<B, WebSocket>,
    ) -> Result<(), CollectiveError> {
        let local_strategy = config.local_all_reduce_strategy;
        let global_strategy = config.global_all_reduce_strategy;

        // Get corresponding devices for each peer
        let devices = tensors
            .iter()
            .map(|(id, tensor)| (*id, B::float_device(tensor)))
            .collect::<HashMap<PeerId, B::Device>>();

        // For Centralized and Tree, we only need to do a reduce here, we'll do a broadcast later
        let main_device = *tensors.keys().next().unwrap();
        let mut tensors_to_reduce = core::mem::take(tensors);
        let mut main_tensor = match local_strategy {
            AllReduceStrategy::Centralized => {
                reduce_sum_centralized::<B>(tensors_to_reduce, &main_device)
            }
            AllReduceStrategy::Tree(arity) => {
                reduce_sum_tree::<B>(tensors_to_reduce, &main_device, arity)
            }
            AllReduceStrategy::Ring => {
                all_reduce_sum_ring::<B>(&mut tensors_to_reduce);
                tensors_to_reduce.remove(&main_device).unwrap()
            }
        };

        // Do aggregation on global level with the main tensor
        main_tensor = global_client
            .all_reduce(main_tensor, global_strategy.unwrap(), op)
            .await
            .map_err(CollectiveError::Global)?;

        // Broadcast result to all devices
        *tensors = match local_strategy {
            AllReduceStrategy::Tree(arity) => {
                broadcast_tree::<B>(devices, main_device, main_tensor, arity)
            }
            // If we chose the ring strategy and we must still broadcast the global result,
            // we use the centralized strategy for broadcasting, but the tree may be better.
            AllReduceStrategy::Centralized | AllReduceStrategy::Ring => {
                broadcast_centralized::<B>(devices, main_device, main_tensor)
            }
        };

        Ok(())
    }

    /// Send a collective error as result to operation caller
    pub fn send_err_to_all(&mut self, err: CollectiveError) {
        self.calls.drain(..).for_each(|op| {
            op.result_sender.send(Err(err.clone())).unwrap();
        });
    }
}
