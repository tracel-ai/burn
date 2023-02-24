use burn_tensor::backend::Backend;

use crate::{grads::Gradients, tensor::ADTensor};

use super::traversal::{BreadthFirstSearch, GraphTraversal};

pub fn backward<B: Backend, const D: usize>(root: ADTensor<B, D>) -> Gradients<B> {
    let mut grads = Gradients::<B>::new();
    let order = root.metadata.order;
    let ops_map = root.map.clone();
    let mut tape = Vec::with_capacity(order);
    for _ in 0..order {
        tape.push(Vec::with_capacity(1));
    }

    let traversal = BreadthFirstSearch::new(&root.metadata, ops_map.clone());

    if let Some(ops) = ops_map.pop(&root.metadata.id) {
        grads.update(
            root.clone(),
            B::ones(B::shape(&root.primitive), &B::device(&root.primitive)),
        );
        ops.backward(&mut grads);
    }

    traversal.traverse(|node| {
        println!("{:?}", node);
        let order = node.order;

        if order == 0 {
            return;
        }
        if let Some(nodes) = tape.get_mut(order) {
            nodes.push(node)
        };
    });
    println!("{:?}", tape);

    for i in (1..order).rev() {
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
