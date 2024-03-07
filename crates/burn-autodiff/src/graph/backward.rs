use burn_tensor::backend::Backend;

use crate::{checkpoint::base::Checkpointer, grads::Gradients, tensor::AutodiffTensor};

use super::{traversal::BreadthFirstSearch, Graph, NodeRef, StepBoxed};

pub fn backward<B: Backend, const D: usize>(root: AutodiffTensor<B, D>) -> Gradients<B::DynTensorPrimitive> {
    let grads = Gradients::new::<B, D>(root.node.clone(), root.primitive);
    let checkpointer = root.graph.build_checkpointer();

    let tape = build_tape(root.node, root.graph);

    execute_steps(tape, grads, checkpointer)
}

fn build_tape<P>(root: NodeRef, graph: Graph<P>) -> Vec<Vec<StepBoxed<P>>> {
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

fn execute_steps<P>(
    tape: Vec<Vec<StepBoxed<P>>>,
    mut grads: Gradients<P>,
    mut checkpointer: Checkpointer,
) -> Gradients<P> {
    tape.into_iter().rev().for_each(|steps| {
        steps
            .into_iter()
            .for_each(|step| step.step(&mut grads, &mut checkpointer))
    });

    #[cfg(feature = "export_tests")]
    // For checkpointing tests
    assert!(checkpointer.is_empty());

    grads
}
