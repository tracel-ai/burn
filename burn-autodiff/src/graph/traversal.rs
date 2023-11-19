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
        let mut steps = graph.steps();
        let root_step = steps
            .remove(&root.id)
            .expect("Root node should have a step registered, did you forget to call `Tensor::register_grad` on the tensor where you need gradients?");

        visited.insert(root.id.clone());
        parents.append(&mut root.parents.clone());
        callback(root, root_step);

        while let Some(id) = parents.pop() {
            let step = match steps.remove(&id) {
                Some(step) => step,
                None => continue,
            };

            let node = step.node();

            if visited.contains(&node.id) {
                continue;
            }

            visited.insert(node.id.clone());

            for id in node.parents.iter() {
                if !visited.contains(id) {
                    parents.push(id.clone());
                }
            }

            callback(node, step);
        }
    }
}
