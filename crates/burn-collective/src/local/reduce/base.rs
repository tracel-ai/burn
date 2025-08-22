use std::{collections::HashMap, sync::mpsc::SyncSender};

use burn_communication::websocket::WebSocket;
use burn_tensor::{ElementConversion, Shape, TensorMetadata, backend::Backend};

use crate::{
    CollectiveConfig, CollectiveError, PeerId, ReduceOperation, ReduceStrategy,
    local::{reduce_sum_centralized, reduce_sum_tree},
    node::base::Node,
};

/// An on-going reduce operation
pub struct ReduceOp<B: Backend> {
    /// reduce calls, one for each calling device
    calls: Vec<ReduceOpCall<B>>,
    /// The reduce operation, as defined by the first caller
    op: ReduceOperation,
    /// The peer that receives the reduce result, as defined by the first caller
    root: PeerId,
    /// The shape of the tensor to reduce, as defined by the first caller
    shape: Shape,
}

/// Struct for each device that calls an reduce operation
pub struct ReduceOpCall<B: Backend> {
    /// Id of the caller of the operation
    caller: PeerId,
    /// The tensor primitive passed as input
    input: B::FloatTensorPrimitive,
    /// Callback for the result of the reduce
    result_sender: SyncSender<ReduceResult<B::FloatTensorPrimitive>>,
}

impl<B: Backend> ReduceOpCall<B> {
    pub fn new(
        caller: PeerId,
        input: B::FloatTensorPrimitive,
        result_sender: SyncSender<ReduceResult<B::FloatTensorPrimitive>>,
    ) -> Self {
        Self {
            caller,
            input,
            result_sender,
        }
    }
}

/// Type sent to the collective client upon completion of a reduce aggregation
pub(crate) type ReduceResult<T> = Result<Option<T>, CollectiveError>;

impl<B: Backend> ReduceOp<B> {
    pub fn new(shape: Shape, reduce_op: ReduceOperation, root: PeerId) -> Self {
        Self {
            calls: vec![],
            op: reduce_op,
            root,
            shape,
        }
    }

    /// Register a call to reduce in this operation.
    /// When the last caller registers a reduce, the operation is executed.
    pub async fn register_call(
        &mut self,
        call: ReduceOpCall<B>,
        op: ReduceOperation,
        root: PeerId,
        peers: &[PeerId],
        config: &CollectiveConfig,
        global_client: &mut Option<Node<B, WebSocket>>,
    ) {
        if self.shape != call.input.shape() {
            self.send_err_to_all(CollectiveError::ReduceShapeMismatch);
            return;
        }
        if self.op != op {
            self.send_err_to_all(CollectiveError::ReduceOperationMismatch);
            return;
        }
        if self.root != root {
            self.send_err_to_all(CollectiveError::ReduceRootMismatch);
            return;
        }

        self.calls.push(call);

        let tensor_count = self.calls.len();
        if tensor_count == peers.len() {
            // Do reduce
            match self.reduce(config, global_client).await {
                Ok(mut result) => {
                    // Return resulting tensor to root, None to others
                    self.calls.drain(..).for_each(|op| {
                        let msg = if op.caller == root {
                            Ok(result.take())
                        } else {
                            Ok(None)
                        };
                        op.result_sender.send(msg).unwrap();
                    });
                }
                Err(err) => {
                    self.send_err_to_all(err);
                }
            }
        }
    }

    async fn reduce(
        &mut self,
        config: &CollectiveConfig,
        global_client: &mut Option<Node<B, WebSocket>>,
    ) -> Result<Option<B::FloatTensorPrimitive>, CollectiveError> {
        let mut tensors = HashMap::new();
        for op in &self.calls {
            tensors.insert(op.caller, op.input.clone());
        }
        let tensor_count = tensors.len() as f32;

        let local_strategy = config.local_reduce_strategy;

        // For Centralized and Tree, we only need to do a reduce here, we'll do a broadcast later
        let tensors_to_reduce = core::mem::take(&mut tensors);
        let mut result = match local_strategy {
            ReduceStrategy::Centralized => {
                reduce_sum_centralized::<B>(tensors_to_reduce, &self.root)
            }
            ReduceStrategy::Tree(arity) => {
                reduce_sum_tree::<B>(tensors_to_reduce, &self.root, arity)
            }
        };

        // Do aggregation on global level with the main tensor
        let result = if let Some(global_client) = global_client {
            let global_strategy = config.global_reduce_strategy.unwrap();
            global_client
                .reduce(result, global_strategy, self.root, self.op)
                .await
                .map_err(CollectiveError::Global)?
        } else {
            // Mean division locally
            if self.op == ReduceOperation::Mean {
                result = B::float_div_scalar(result, tensor_count.elem())
            }
            Some(result)
        };

        Ok(result)
    }

    /// Send a collective error as result to operation caller
    fn send_err_to_all(&mut self, err: CollectiveError) {
        self.calls.drain(..).for_each(|op| {
            op.result_sender.send(Err(err.clone())).unwrap();
        });
    }
}
