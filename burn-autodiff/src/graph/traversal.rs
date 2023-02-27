use std::collections::HashSet;

use super::{Graph, NodeRef, StepBoxed};

/// Breadth for search algorithm.
pub struct BreadthFirstSearch;

impl BreadthFirstSearch {
    /// Traverse the graph of backward steps from a root node.
    pub fn traverse<F: FnMut(NodeRef, StepBoxed)>(
        &self,
        root: NodeRef,
        graph: Graph,
        mut callback: F,
    ) {
        let mut visited = HashSet::with_capacity(root.order);
        let mut parents = Vec::with_capacity(root.order);

        visited.insert(root.id.clone());
        parents.append(&mut root.parents.clone());

        let mut steps = graph.steps();
        let root_step = steps.remove(&root.id).unwrap();
        callback(root, root_step);

        while let Some(id) = parents.pop() {
            let step = match steps.remove(&id) {
                Some(step) => step,
                None => continue,
            };

            let node = step.node();

            let id = &node.id;
            if visited.contains(id) {
                continue;
            }

            visited.insert(id.clone());

            for id in node.parents.iter() {
                if !visited.contains(id) {
                    parents.push(id.clone());
                }
            }

            callback(node, step);
        }
    }
}
