use crate::graph::{node::BackwardNode, ops::RecordedOpsParentRef};
use std::collections::HashSet;

pub trait GraphTraversal {
    fn traverse<F: FnMut(RecordedOpsParentRef)>(&self, callback: F);
}

#[derive(new)]
pub struct BreadthFirstSearch<'a, T> {
    node: &'a BackwardNode<T>,
}

impl<'a, T> GraphTraversal for BreadthFirstSearch<'a, T> {
    fn traverse<F: FnMut(RecordedOpsParentRef)>(&self, mut callback: F) {
        let mut visited = HashSet::with_capacity(self.node.order);
        let mut parents = Vec::with_capacity(self.node.order);

        visited.insert(self.node.id.clone());
        parents.append(&mut self.node.ops.backward_parents());

        loop {
            let node = match parents.pop() {
                Some(node) => node,
                None => break,
            };

            let id = node.id();
            visited.insert(id.clone());

            for parent in node.backward_parents() {
                let id = parent.id();

                if !visited.contains(id) {
                    parents.push(parent);
                }
            }

            callback(node);
        }
    }
}
