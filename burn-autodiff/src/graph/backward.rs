use burn_tensor::backend::Backend;

use crate::{grads::Gradients, tensor::ADTensor};

use super::{traversal::BreadthFirstSearch, Graph, NodeRef, StepBoxed};

pub fn backward<B: Backend, const D: usize>(root: ADTensor<B, D>) -> Gradients<B> {
    let grads = init_grads(root.node.clone(), root.primitive);
    let tape = build_tape(root.node, root.graph);

    execute_steps(tape, grads)
}

fn init_grads<B: Backend, const D: usize>(
    root_node: NodeRef,
    root_tensor: B::TensorPrimitive<D>,
) -> Gradients<B> {
    let mut grads = Gradients::<B>::new();
    grads.update(
        root_node.clone(),
        B::ones(B::shape(&root_tensor), &B::device(&root_tensor)),
    );

    grads
}

fn build_tape<B: Backend>(root: NodeRef, graph: Graph<B>) -> Vec<Vec<StepBoxed<B>>> {
    let mut tape = Vec::with_capacity(root.order);
    for _ in 0..root.order {
        tape.push(Vec::with_capacity(1));
    }

    BreadthFirstSearch.traverse(root, graph, |node, step| {
        if node.order == 0 {
            return;
        }

        if let Some(steps) = tape.get_mut(node.order - 1) {
            steps.push(step)
        };
    });

    tape
}

fn execute_steps<B: Backend>(
    mut tape: Vec<Vec<StepBoxed<B>>>,
    mut grads: Gradients<B>,
) -> Gradients<B> {
    for i in (0..tape.len()).rev() {
        let steps = match tape.get_mut(i) {
            Some(val) => val,
            None => continue,
        };

        // Take ownership of steps.
        let mut empty = Vec::new();
        std::mem::swap(steps, &mut empty);

        for step in empty {
            step.step(&mut grads);
        }
    }

    grads
}
