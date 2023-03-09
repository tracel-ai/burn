use burn_tensor::backend::Backend;

use crate::{grads::Gradients, tensor::ADTensor};

use super::{traversal::BreadthFirstSearch, Graph, NodeRef, StepBoxed};

pub fn backward<B: Backend, const D: usize>(root: ADTensor<B, D>) -> Gradients {
    let grads = Gradients::new::<B, D>(root.node.clone(), root.primitive);
    let tape = build_tape(root.node, root.graph);

    execute_steps(tape, grads)
}

fn build_tape(root: NodeRef, graph: Graph) -> Vec<Vec<StepBoxed>> {
    let mut tape = (0..root.order)
        .map(|_| Vec::with_capacity(1))
        .collect::<Vec<_>>();

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

fn execute_steps(tape: Vec<Vec<StepBoxed>>, mut grads: Gradients) -> Gradients {
    tape.into_iter()
        .rev()
        .for_each(|steps| steps.into_iter().for_each(|step| step.step(&mut grads)));
    grads
}
