use super::{Step, StepBoxed};
use crate::NodeID;
use std::collections::{HashMap, HashSet};

/// Breadth for search algorithm.
pub struct BreadthFirstSearch;

pub trait TraversalItem {
    fn id(&self) -> NodeID;
    fn parents(&self) -> Vec<NodeID>;
}

impl BreadthFirstSearch {
    /// Traverse the graph of backward steps from a root node.
    pub fn traverse<F, I>(
        &self,
        root_id: NodeID,
        root_step: I,
        steps: &mut HashMap<NodeID, I>,
        mut callback: F,
    ) where
        F: FnMut(NodeID, I),
        I: TraversalItem,
    {
        let mut visited = HashSet::new();
        let mut parents = Vec::new();

        visited.insert(root_id);
        parents.append(&mut root_step.parents());

        callback(root_id, root_step);

        while let Some(id) = parents.pop() {
            let step = match steps.remove(&id) {
                Some(step) => step,
                None => continue,
            };

            let step_node = step.id();
            let step_parents = step.parents();

            if visited.contains(&step_node) {
                continue;
            }

            visited.insert(step_node);

            for id in step_parents.iter() {
                if !visited.contains(id) {
                    parents.push(*id);
                }
            }

            callback(step_node, step);
        }
    }
}

impl TraversalItem for StepBoxed {
    fn id(&self) -> NodeID {
        Step::node(self.as_ref())
    }

    fn parents(&self) -> Vec<NodeID> {
        Step::parents(self.as_ref())
    }
}
