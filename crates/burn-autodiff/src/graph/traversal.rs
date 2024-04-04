use std::collections::{HashMap, HashSet};

use crate::NodeID;

use super::StepBoxed;

/// Breadth for search algorithm.
pub struct BreadthFirstSearch;

impl BreadthFirstSearch {
    /// Traverse the graph of backward steps from a root node.
    pub fn traverse<F: FnMut(NodeID, StepBoxed)>(
        &self,
        root_id: NodeID,
        root_step: StepBoxed,
        steps: &mut HashMap<NodeID, StepBoxed>,
        mut callback: F,
    ) {
        let root_order = root_step.order();
        let mut visited = HashSet::with_capacity(root_order);
        let mut parents = Vec::with_capacity(root_order);

        visited.insert(root_id);
        parents.append(&mut root_step.parents());
        callback(root_id, root_step);

        while let Some(id) = parents.pop() {
            let step = match steps.remove(&id) {
                Some(step) => step,
                None => continue,
            };

            let step_node = step.node();
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
