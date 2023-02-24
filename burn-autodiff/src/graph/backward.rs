use burn_tensor::backend::Backend;

use crate::grads::Gradients;

use super::{
    ops::{OpsMap, OpsMetadataRef},
    traversal::{BreadthFirstSearch, GraphTraversal},
};

pub fn backward<B: Backend>(root: OpsMetadataRef, ops_map: OpsMap<B>) -> Gradients<B> {
    let mut grads = Gradients::<B>::new();
    let mut tape = Vec::with_capacity(root.order);
    for _ in 0..root.order {
        tape.push(Vec::with_capacity(1));
    }

    let traversal = BreadthFirstSearch::new(&root, ops_map.clone());

    traversal.traverse(|node| {
        let order = node.order;

        if order == 0 {
            return;
        }
        if let Some(nodes) = tape.get_mut(order) {
            nodes.push(node)
        };
    });

    for i in (1..root.order).rev() {
        let nodes = match tape.get(i) {
            Some(nodes) => nodes,
            None => continue,
        };

        for node in nodes {
            if let Some(ops) = ops_map.pop(&node.id) {
                ops.backward(&mut grads);
            }
        }
    }

    grads
}
