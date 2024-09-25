use crate::{
    checkpoint::builder::CheckpointerBuilder,
    grads::Gradients,
    graph::StepBoxed,
    tensor::{AutodiffTensor, NodeRefCount},
};
use burn_tensor::backend::Backend;

/// Client used to communicate with the autodiff server.
pub trait AutodiffClient: Send + Clone {
    /// Register a new step.
    fn register(&self, node_id: NodeRefCount, step: StepBoxed, actions: CheckpointerBuilder);
    /// Call backpropagation from the given tensor.
    fn backward<B: Backend>(&self, tensor: AutodiffTensor<B>) -> Gradients;
}

/// Client implementation in used.
#[cfg(feature = "async")]
pub type AutodiffClientImpl = super::mspc::ChannelClient;

/// Client implementation in used.
#[cfg(not(feature = "async"))]
pub type AutodiffClientImpl = super::mutex::MutexClient;
