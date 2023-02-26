use burn_tensor::backend::Backend;

use crate::{grads::Gradients, tensor::ADTensor};

use super::traversal::BreadthFirstSearch;

pub fn backward<B: Backend, const D: usize>(root: ADTensor<B, D>) -> Gradients<B> {
    let mut grads = Gradients::<B>::new();
    let root_order = root.metadata.order;
    let root_metadata = root.metadata;
    let root_primitive = root.primitive;
    let root_graph = root.graph;

    let mut ops_map = root_graph.extract();

    let mut tape = Vec::with_capacity(root_order);
    for _ in 0..root_order {
        tape.push(Vec::with_capacity(1));
    }

    if let Some(ops) = ops_map.remove(&root_metadata.id) {
        grads.update(
            root_metadata.clone(),
            B::ones(B::shape(&root_primitive), &B::device(&root_primitive)),
        );
        ops.backward(&mut grads);
    }

    BreadthFirstSearch.traverse(root_metadata, &ops_map, |node| {
        if node.order == 0 {
            return;
        }

        if let Some(nodes) = tape.get_mut(node.order) {
            nodes.push(node)
        };
    });

    for i in (1..root_order).rev() {
        let nodes = match tape.get(i) {
            Some(nodes) => nodes,
            None => continue,
        };

        for node in nodes {
            if let Some(ops) = ops_map.remove(&node.id) {
                ops.backward(&mut grads);
            }
        }
    }

    grads
}
