use burn_tensor::backend::Backend;

use crate::{grads::Gradients, tensor::ADTensor};

use super::{traversal::BreadthFirstSearch, Graph, NodeRef, StepBoxed};

pub fn backward<B: Backend, const D: usize>(root: ADTensor<B, D>) -> Gradients {
    let grads = Gradients::new::<B, D>(root.node.clone(), root.primitive);
    let tape = build_tape(root.node, root.graph);

    execute_steps(tape, grads)
}

fn build_tape(root: NodeRef, graph: Graph) -> Vec<Vec<StepBoxed>> {
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

fn execute_steps(mut tape: Vec<Vec<StepBoxed>>, mut grads: Gradients) -> Gradients {
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
