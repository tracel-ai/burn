use crate::global::node::base::Node;
use crate::local::tensor_map::CollectiveTensorMap;
use crate::{CollectiveConfig, CollectiveError, PeerId, ReduceOperation, local};
use burn_communication::websocket::WebSocket;
use burn_std::Shape;
use burn_tensor::TensorMetadata;
use burn_tensor::backend::Backend;
use std::collections::HashMap;
use std::sync::mpsc::SyncSender;

/// An on-going all-reduce operation
#[derive(Debug)]
pub struct AllReduceOp<B: Backend> {
    /// all-reduce calls, one for each calling device
    calls: Vec<AllReduceOpCall<B>>,
    /// The reduce operation of the current all-reduce, as defined by the first caller
    op: ReduceOperation,
    /// The shape of the current all-reduce, as defined by the first caller
    shape: Shape,
}

/// Struct for each device that calls an all-reduce operation
#[derive(Debug)]
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

    fn peers(&self) -> Vec<PeerId> {
        self.calls.iter().map(|c| c.caller).collect()
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
    #[tracing::instrument(
        skip(self, config, global_client),
        fields(
            ?self.op,
            ?self.shape,
            self.peers = ?self.peers(),
        )
    )]
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
    #[tracing::instrument(skip(self, config, global_client))]
    async fn all_reduce(
        &mut self,
        config: &CollectiveConfig,
        global_client: &mut Option<Node<B, WebSocket>>,
    ) -> Result<CollectiveTensorMap<B>, CollectiveError> {
        let mut tensors = HashMap::new();
        for call in &self.calls {
            tensors.insert(call.caller, call.input.clone());
        }

        let op = self.op;
        Ok(if let Some(global_client) = global_client.as_mut() {
            local::all_reduce_with_global::<B>(tensors, op, config, global_client).await?
        } else {
            local::all_reduce_local_only::<B>(tensors, op, config).await?
        })
    }

    /// Send a collective error as result to operation caller
    pub fn send_err_to_all(&mut self, err: CollectiveError) {
        self.calls.drain(..).for_each(|op| {
            op.result_sender.send(Err(err.clone())).unwrap();
        });
    }
}
