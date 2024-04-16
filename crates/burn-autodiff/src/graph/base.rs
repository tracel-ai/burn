use super::NodeID;
use crate::{checkpoint::base::Checkpointer, grads::Gradients};
use std::collections::HashMap;

/// Backward step for reverse mode autodiff.
pub trait Step: Send + std::fmt::Debug {
    /// Executes the step and consumes it.
    fn step(self: Box<Self>, grads: &mut Gradients, checkpointer: &mut Checkpointer);
    /// The node associated to the step.
    fn node(&self) -> NodeID;
    /// The parents of the node associated to the step.
    fn parents(&self) -> Vec<NodeID>;
    fn order(&self) -> usize;
}

pub type StepBoxed = Box<dyn Step>;
pub type NodeSteps = HashMap<NodeID, StepBoxed>;
