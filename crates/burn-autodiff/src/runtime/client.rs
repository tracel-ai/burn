use crate::{
    checkpoint::builder::CheckpointerBuilder,
    grads::Gradients,
    graph::StepBoxed,
    tensor::{AutodiffTensor, NodeRefCount},
};
#[cfg(not(feature = "distributed"))]
use burn_backend::Backend;
#[cfg(feature = "distributed")]
use burn_backend::distributed::DistributedBackend;

/// Client used to communicate with the autodiff server.
pub trait AutodiffClient: Send + Clone {
    /// Register a new step.
    fn register(&self, node_id: NodeRefCount, step: StepBoxed, actions: CheckpointerBuilder);
    #[cfg(not(feature = "distributed"))]
    /// Call backpropagation from the given tensor.
    fn backward<B: Backend>(&self, tensor: AutodiffTensor<B>) -> Gradients;
    #[cfg(feature = "distributed")]
    /// Call backpropagation from the given tensor.
    fn backward<B: DistributedBackend>(&self, tensor: AutodiffTensor<B>) -> Gradients;
}

/// Client implementation in used.
pub type AutodiffClientImpl = super::graph::GraphMutexClient;
