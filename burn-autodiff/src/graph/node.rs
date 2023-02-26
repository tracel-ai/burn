use std::sync::Arc;

use burn_common::id::IdGenerator;

use super::Requirement;

/// Node in the autodiff graph.
///
/// # Notes
///
/// Nodes contain graph metadata and should be used wrapped in an Arc for cheap cloning.
#[derive(new, Debug)]
pub struct Node {
    pub parents: Vec<NodeID>,
    pub order: usize,
    pub id: NodeID,
    pub requirement: Requirement,
}
/// Read only [node](Node) reference cheap to clone.
pub type NodeRef = Arc<Node>;

impl Node {
    /// Returns the [node](Node) only if gradients are required.
    pub fn clone_if_require_grad(self: &Arc<Self>) -> Option<NodeRef> {
        if self.requirement.is_none() {
            return None;
        }

        Some(self.clone())
    }
}

/// Unique identifier generated for each [node](Node).
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct NodeID {
    pub(crate) value: String,
}

impl NodeID {
    /// Create a unique [node id](NodeID).
    pub fn new() -> Self {
        Self {
            value: IdGenerator::generate(),
        }
    }
}

impl Default for NodeID {
    fn default() -> Self {
        Self::new()
    }
}
