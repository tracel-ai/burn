use crate::{
    checkpoint::{base::Checkpointer, builder::CheckpointerBuilder},
    grads::Gradients,
    graph::{traversal::BreadthFirstSearch, StepBoxed},
    tensor::AutodiffTensor,
    NodeID,
};
use burn_tensor::backend::Backend;
use std::collections::HashMap;

#[derive(Default)]
pub struct AutodiffServer {
    steps: HashMap<NodeID, StepBoxed>,
    actions_builder: HashMap<NodeID, CheckpointerBuilder>,
}

impl AutodiffServer {
    pub fn register(&mut self, node: NodeID, step: StepBoxed, actions: CheckpointerBuilder) {
        self.steps.insert(node.clone(), step);
        self.actions_builder.insert(node, actions);
    }

    pub fn backward<B: Backend, const D: usize>(
        &mut self,
        root: AutodiffTensor<B, D>,
    ) -> Gradients {
        let step = self.steps.remove(&root.node.id).expect(
            "Root node should have a step registered, did you forget to call \
             `Tensor::register_grad` on the tensor where you need gradients?",
        );
        let builder = self.actions_builder.remove(&root.node.id).unwrap();

        let grads = Gradients::new::<B, D>(root.node.clone(), root.primitive);
        let (tape, builder) = self.build_tape(root.node.id.clone(), step, builder);
        let checkpointer = builder.build(&self.steps);

        Self::execute_steps(tape, grads, checkpointer)
    }

    fn build_tape(
        &mut self,
        root: NodeID,
        root_step: StepBoxed,
        mut builder: CheckpointerBuilder,
    ) -> (Vec<Vec<StepBoxed>>, CheckpointerBuilder) {
        let mut tape = (0..root_step.order())
            .map(|_| Vec::with_capacity(1))
            .collect::<Vec<_>>();

        BreadthFirstSearch.traverse(root, root_step, &mut self.steps, |id, step| {
            let order = step.order();
            if order == 0 {
                return;
            }

            if let Some(steps) = tape.get_mut(order - 1) {
                steps.push(step);
            }

            if let Some(node_builder) = self.actions_builder.remove(&id) {
                builder.extend(node_builder);
            }
        });

        (tape, builder)
    }

    fn execute_steps(
        tape: Vec<Vec<StepBoxed>>,
        mut grads: Gradients,
        mut checkpointer: Checkpointer,
    ) -> Gradients {
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

    pub fn drop_node(&self, node_id: NodeID) {
        //todo
    }
}
