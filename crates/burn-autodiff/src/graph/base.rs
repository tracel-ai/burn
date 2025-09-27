use super::NodeId;
use crate::{checkpoint::base::Checkpointer, grads::Gradients, graph::Parent};
use alloc::boxed::Box;

/// Backward step for reverse mode autodiff.
pub trait Step: Send + core::fmt::Debug {
    /// Executes the step and consumes it.
    fn step(self: Box<Self>, grads: &mut Gradients, checkpointer: &mut Checkpointer);
    /// Depth of the operation relative to the first node added to a graph.
    fn depth(&self) -> usize;
    /// The node associated to the step.
    fn node(&self) -> NodeId;
    /// The parents of the node associated to the step.
    fn parents(&self) -> &[Parent];
}

pub type StepBoxed = Box<dyn Step>;
