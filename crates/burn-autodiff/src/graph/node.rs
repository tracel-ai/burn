use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::checkpoint::retro_forward::RetroForward;
use crate::runtime::AutodiffClientImpl;

use super::Requirement;

#[derive(Debug, Clone)]
pub enum ComputingProperty {
    ComputeBound,
    MemoryBound {
        retro_forward: Arc<dyn RetroForward>,
    },
    Ambiguous, // Maybe autotune someday
}

/// This is safe only because we only call RetroForward on the autodiff server.
/// Therefore, the trait will never be used by multiple threads at the same time.
///
/// TODO: Find a way to avoid cloning the compute property, which will remove the need to add the
/// Arc, which will make (dyn RetroForward) safely implement Send.
unsafe impl Send for ComputingProperty {}
/// unsafe Sync is required because Send is only implemented for Arc<Sync>, not Arc<Send>.
unsafe impl Sync for ComputingProperty {}

/// A node contains graph metadata and should be used wrapped in an Arc for cheap cloning.
#[derive(new, Debug)]
pub struct Node {
    pub parents: Vec<NodeID>,
    pub order: usize,
    pub id: NodeID,
    pub requirement: Requirement,
    pub properties: ComputingProperty,
    pub client: AutodiffClientImpl,
}
pub type NodeRef = Arc<Node>;

impl Node {
    /// Returns the [node](Node) only if gradients are required.
    pub fn clone_if_require_grad(self: &Arc<Self>) -> Option<NodeRef> {
        match self.requirement.is_none() {
            true => None,
            false => Some(self.clone()),
        }
    }
}

/// Unique identifier generated for each node.
#[derive(Clone, Hash, PartialEq, Eq, Debug, Copy)]
pub struct NodeID {
    /// The integer representation of the id
    pub value: u64,
}

impl NodeID {
    /// Create a unique [node id](NodeID).
    pub fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let value = COUNTER.fetch_add(1, Ordering::Relaxed);
        if value == u64::MAX {
            panic!("NodeID overflowed");
        }
        Self { value }
    }
}

impl Default for NodeID {
    fn default() -> Self {
        Self::new()
    }
}
