use crate::{
    checkpoint::builder::CheckpointerBuilder,
    grads::Gradients,
    graph::StepBoxed,
    tensor::{AutodiffTensor, NodeRefCount},
};
use burn_tensor::backend::Backend;

pub trait AutodiffClient: Send + Clone {
    fn register(&self, node_id: NodeRefCount, ops: StepBoxed, actions: CheckpointerBuilder);
    fn backward<B: Backend, const D: usize>(&self, root: AutodiffTensor<B, D>) -> Gradients;
}

#[cfg(feature = "std")]
pub type AutodiffClientImpl = super::mspc::ChannelClient;

#[cfg(not(feature = "std"))]
pub type AutodiffClientImpl = super::mutex::MutexClient;
