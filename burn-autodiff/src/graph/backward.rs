use burn_tensor::backend::Backend;

use crate::{grads::Gradients, tensor::ADTensor};

use super::traversal::BreadthFirstSearch;

pub fn backward<B: Backend, const D: usize>(root: ADTensor<B, D>) -> Gradients<B> {
    let mut grads = Gradients::<B>::new();
    let root_order = root.node.order;
    let root_node = root.node;
    // let root_primitive = root.primitive;
    let root_graph = root.graph;

    let mut tape = Vec::with_capacity(root_order);
    for _ in 0..root_order {
        tape.push(Vec::with_capacity(1));
    }

    // if let Some(ops) = ops_map.remove(&root_node.id) {
    //     grads.update(
    //         root_node.clone(),
    //         B::ones(B::shape(&root_primitive), &B::device(&root_primitive)),
    //     );
    //     ops.step(&mut grads);
    // }

    BreadthFirstSearch.traverse(root_node, root_graph, |node, step| {
        if node.order == 0 {
            return;
        }

        if let Some(steps) = tape.get_mut(node.order) {
            steps.push(step)
        };
    });

    for i in (1..root_order).rev() {
        let steps = match tape.get_mut(i) {
            Some(val) => val,
            None => continue,
        };
        let mut empty = Vec::new();
        std::mem::swap(steps, &mut empty);

        for step in empty {
            step.step(&mut grads);
        }
    }

    grads
}
