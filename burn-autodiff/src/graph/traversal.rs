use std::collections::{HashMap, HashSet};

use burn_tensor::backend::Backend;

use super::ops::{MetadataRef, Node, OpsID};

pub struct BreadthFirstSearch;

impl BreadthFirstSearch {
    pub fn traverse<B: Backend, F: FnMut(MetadataRef)>(
        &self,
        root: MetadataRef,
        ops: &HashMap<OpsID, Node<B>>,
        mut callback: F,
    ) {
        let mut visited = HashSet::with_capacity(root.order);
        let mut parents = Vec::with_capacity(root.order);

        visited.insert(root.id.clone());
        parents.append(&mut root.parents.clone());

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
