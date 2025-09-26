use crate::{
    checkpoint::builder::CheckpointerBuilder,
    grads::Gradients,
    graph::StepBoxed,
    tensor::{AutodiffTensor, NodeRefCount},
};
use burn_common::id::StreamId;
use burn_tensor::backend::Backend;

/// Client used to communicate with the autodiff server.
pub trait AutodiffClient: Send + Clone {
    /// Register a new step.
    fn register(
        &self,
        stream_id: StreamId,
        node_id: NodeRefCount,
        step: StepBoxed,
        actions: CheckpointerBuilder,
    );
    /// Call backpropagation from the given tensor.
    fn backward<B: Backend>(&self, tensor: AutodiffTensor<B>) -> Gradients;
}

/// Client implementation in used.
// pub type AutodiffClientImpl = super::mutex::tmp::MutexClient;
pub type AutodiffClientImpl = super::mutex::MultiStreamMutexClient;
