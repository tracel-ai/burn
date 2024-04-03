use crate::{
    checkpoint::{base::Checkpointer, builder::CheckpointerBuilder},
    grads::Gradients,
    graph::{traversal::BreadthFirstSearch, StepBoxed},
    tensor::{AutodiffTensor, NodeRefCount},
    NodeID,
};
use burn_tensor::backend::Backend;
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

#[derive(Hash, PartialEq, Eq)]
struct GraphId {
    node: NodeID,
}

#[derive(Default)]
pub struct AutodiffServer {
    steps: HashMap<NodeID, StepBoxed>,
    actions_builder: HashMap<NodeID, CheckpointerBuilder>,
    graphs: HashMap<GraphId, HashSet<NodeRefCount>>,
}

impl AutodiffServer {
    pub fn register(&mut self, rc: NodeRefCount, step: StepBoxed, actions: CheckpointerBuilder) {
        let parents = step.parents();
        let node_id = *rc.as_ref();

        self.steps.insert(node_id, step);
        self.actions_builder.insert(*rc.as_ref(), actions);
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

        let gradients = Self::execute_steps(tape, grads, checkpointer);
        self.free_detach();
        gradients
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
                return Action::Continue;
            }

            if let Some(steps) = tape.get_mut(order - 1) {
                steps.push(step);
            }

            if let Some(node_builder) = self.actions_builder.remove(&id) {
                builder.extend(node_builder);
            }

            Action::Continue
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

    fn free_graph(&mut self, node_id: NodeID) {
        let step = self.steps.remove(&node_id).unwrap();

        BreadthFirstSearch.traverse(node_id, step, &mut self.steps, |id, _step| {
            self.actions_builder.remove(&id);
            Action::Continue
        })
    }

    fn free_detach(&mut self) {
        let mut should_free = Vec::new();
        for (node, nodes) in self.graphs.iter() {
            let rc = nodes.get(node).unwrap();

            if Arc::strong_count(rc) == 1 {
                should_free.push(*node);
            }
        }

        for node in should_free {
            self.free_graph(node);
        }
    }
}
