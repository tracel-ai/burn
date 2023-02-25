use std::collections::{HashMap, HashSet};

use burn_tensor::backend::Backend;

use super::ops::{MetadataRef, Node, OpsID};

pub trait GraphTraversal<B: Backend> {
    fn traverse<F: FnMut(MetadataRef)>(&self, callback: F, ops: &HashMap<OpsID, Node<B>>);
}

#[derive(new)]
pub struct BreadthFirstSearch<'a> {
    node: &'a MetadataRef,
}

impl<'a, B: Backend> GraphTraversal<B> for BreadthFirstSearch<'a> {
    fn traverse<F: FnMut(MetadataRef)>(&self, mut callback: F, ops: &HashMap<OpsID, Node<B>>) {
        let mut visited = HashSet::with_capacity(self.node.order);
        let mut parents = Vec::with_capacity(self.node.order);

        visited.insert(self.node.id.clone());
        parents.append(&mut self.node.parents.clone());

        while let Some(node) = parents.pop() {
            let node = ops.get(&node).map(|node| node.metadata()).unwrap();

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

            callback(node);
        }
    }
}
