use super::{Step, StepBoxed};
use crate::{
    NodeId,
    collections::{HashMap, HashSet},
    graph::Parent,
};
use alloc::{vec, vec::Vec};

/// Breadth for search algorithm.
pub struct GraphTraversal;

pub trait TraversalItem {
    fn parents(&self) -> &[Parent];
}

impl GraphTraversal {
    /// Validate ancestry and collect each available step once, starting with the root.
    ///
    /// A missing leaf step is harmless: gradients terminate there. Every non-leaf must
    /// still have its backward step, even when reached through a newly recorded output.
    /// No steps or checkpoint state are removed until the caller consumes the returned IDs.
    /// The IDs follow traversal order; execution must still respect step depth.
    pub fn collect_steps<I: TraversalItem>(
        &self,
        root_id: NodeId,
        steps: &HashMap<NodeId, I>,
    ) -> Result<Vec<NodeId>, NodeId> {
        let root = steps.get(&root_id).ok_or(root_id)?;
        let mut nodes = vec![root_id];
        let mut visited = HashSet::new();
        let mut parents = root.parents().to_vec();
        visited.insert(root_id);

        while let Some(parent) = parents.pop() {
            if !visited.insert(parent.id) {
                continue;
            }
            match steps.get(&parent.id) {
                Some(step) => {
                    nodes.push(parent.id);
                    parents.extend_from_slice(step.parents());
                }
                None if parent.is_leaf => {}
                None => return Err(parent.id),
            }
        }

        Ok(nodes)
    }
}

impl TraversalItem for StepBoxed {
    fn parents(&self) -> &[Parent] {
        Step::parents(self.as_ref())
    }
}
