use burn_communication::Protocol;
use burn_tensor::{ElementConversion, Shape, TensorMetadata, backend::Backend};
use std::sync::mpsc::SyncSender;

use crate::local::tensor_map::CollectiveTensorMap;
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

    fn peers(&self) -> Vec<PeerId> {
        self.calls.iter().map(|c| c.caller).collect()
    }

    /// Register a call to reduce in this operation.
    /// When the last caller registers a reduce, the operation is executed.
    pub fn register_call(
        &mut self,
        caller: PeerId,
        input: B::FloatTensorPrimitive,
        result_sender: SyncSender<ReduceResult<B::FloatTensorPrimitive>>,
        op: ReduceOperation,
        root: PeerId,
        peer_count: usize,
    ) -> Result<bool, CollectiveError> {
        if self.shape != input.shape() {
            return Err(CollectiveError::ReduceShapeMismatch);
        }
        if self.op != op {
            return Err(CollectiveError::ReduceOperationMismatch);
        }
        if self.root != root {
            return Err(CollectiveError::ReduceRootMismatch);
        }

        self.calls.push(ReduceOpCall {
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
    pub async fn execute<P: Protocol>(
        mut self,
        root: PeerId,
        config: &CollectiveConfig,
        global_client: &mut Option<Node<B, P>>,
    ) {
        match self.reduce(config, global_client).await {
            Ok(mut result) => {
                // Return resulting tensor to root, None to others
                self.calls.into_iter().for_each(|op| {
                    let msg = if op.caller == root {
                        Ok(result.take())
                    } else {
                        Ok(None)
                    };
                    op.result_sender.send(msg).unwrap();
                });
            }
            Err(err) => {
                self.fail(err);
            }
        }
    }

    #[tracing::instrument(skip(self, config, global_client))]
    async fn reduce<P: Protocol>(
        &mut self,
        config: &CollectiveConfig,
        global_client: &mut Option<Node<B, P>>,
    ) -> Result<Option<B::FloatTensorPrimitive>, CollectiveError> {
        let tensors: CollectiveTensorMap<B> = self
            .calls
            .iter()
            .map(|op| (op.caller, op.input.clone()))
            .collect();

        // For Centralized and Tree, we only need to do a reduce here, we'll do a broadcast later
        let mut local_sum = match config.local_reduce_strategy {
            ReduceStrategy::Centralized => reduce_sum_centralized::<B>(tensors, &self.root),
            ReduceStrategy::Tree(arity) => reduce_sum_tree::<B>(tensors, &self.root, arity),
        };

        // Do aggregation on a global level with the main tensor
        let result = if let Some(global_client) = global_client {
            let global_strategy = config.global_reduce_strategy.unwrap();
            global_client
                .reduce(local_sum, global_strategy, self.root, self.op)
                .await
                .map_err(CollectiveError::Global)?
        } else {
            // Mean division locally
            if self.op == ReduceOperation::Mean {
                let local_tensor_count = self.calls.len() as f32;
                local_sum = B::float_div_scalar(local_sum, local_tensor_count.elem())
            }
            Some(local_sum)
        };

        Ok(result)
    }

    /// Send a collective error as result to operation caller
    pub fn fail(self, err: CollectiveError) {
        self.calls.into_iter().for_each(|op| {
            op.result_sender.send(Err(err.clone())).unwrap();
        });
    }
}
