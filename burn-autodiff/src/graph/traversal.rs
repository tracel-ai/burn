use std::collections::HashSet;

use burn_tensor::backend::Backend;

use super::ops::{OpsMap, OpsMetadataRef};

pub trait GraphTraversal<B: Backend> {
    fn traverse<F: FnMut(OpsMetadataRef)>(&self, callback: F);
}

#[derive(new)]
pub struct BreadthFirstSearch<'a, B: Backend> {
    node: &'a OpsMetadataRef,
    ops: OpsMap<B>,
}

impl<'a, B: Backend> GraphTraversal<B> for BreadthFirstSearch<'a, B> {
    fn traverse<F: FnMut(OpsMetadataRef)>(&self, mut callback: F) {
        let mut visited = HashSet::with_capacity(self.node.order);
        let mut parents = Vec::with_capacity(self.node.order);

        visited.insert(self.node.id.clone());
        parents.append(&mut self.node.parents.clone());

        loop {
            let node = match parents.pop() {
                Some(node) => node,
                None => break,
            };

            let node = self.ops.metadata(&node).unwrap();

            let id = &node.id;
            if visited.contains(&id) {
                continue;
            }

            visited.insert(id.clone());

            for id in node.parents.iter() {
                if !visited.contains(&id) {
                    parents.push(id.clone());
                }
            }

            callback(node);
        }
    }
}
