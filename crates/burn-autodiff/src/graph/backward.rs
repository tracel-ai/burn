use burn_tensor::backend::Backend;

use crate::{grads::Gradients, tensor::AutodiffTensor};

use super::{traversal::BreadthFirstSearch, Graph, NodeRef, StepBoxed};

pub fn backward<B: Backend, const D: usize>(root: AutodiffTensor<B, D>) -> Gradients<B> {
    let grads = Gradients::new::<D>(root.node.clone(), root.primitive);
    let tape = build_tape(root.node, root.graph);

    execute_steps(tape, grads)
}

fn build_tape<B: Backend>(root: NodeRef, graph: Graph<B>) -> Vec<Vec<StepBoxed<B>>> {
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

fn execute_steps<B: Backend>(
    tape: Vec<Vec<StepBoxed<B>>>,
    mut grads: Gradients<B>,
) -> Gradients<B> {
    tape.into_iter()
        .rev()
        .for_each(|steps| steps.into_iter().for_each(|step| step.step(&mut grads)));
    grads
}
