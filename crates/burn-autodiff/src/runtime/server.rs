use super::memory_management::GraphMemoryManagement;
use crate::{
    NodeId,
    checkpoint::{
        base::{Checkpointer, NodeTree},
        builder::CheckpointerBuilder,
    },
    collections::HashMap,
    grad_sync::api::get_gradient_sync_client,
    grads::Gradients,
    graph::{
        NodeRef, StepBoxed,
        traversal::{BreadthFirstSearch, TraversalItem},
    },
    tensor::NodeRefCount,
};
use alloc::vec::Vec;
use burn_backend::{Backend, ShardedParams, tensor::FloatTensor};

struct TapeResult {
    tape: Vec<Vec<StepBoxed>>,
    checkpointer: Checkpointer,
    n_required_map: HashMap<NodeId, usize>,
    sharded_params: HashMap<NodeId, ShardedParams>,
}

#[derive(Default)]
pub struct AutodiffServer {
    steps: HashMap<NodeId, StepBoxed>,
    actions_builder: HashMap<NodeId, CheckpointerBuilder>,
    memory_management: GraphMemoryManagement,
}

/// Defines how nodes are clean.
pub trait NodeCleaner {
    /// Initialize a new cleaner.
    fn init() -> Self;
    /// Cleans a single [node](NodeId).
    fn clean(&mut self, node: &NodeId);
}

impl AutodiffServer {
    pub fn extend(&mut self, other: AutodiffServer) {
        self.steps.extend(other.steps);
        self.actions_builder.extend(other.actions_builder);
        self.memory_management.extend(other.memory_management);
    }

    pub fn register(&mut self, rc: NodeRefCount, step: StepBoxed, actions: CheckpointerBuilder) {
        let parents = step.parents();
        let node_id = *rc.as_ref();

        self.memory_management.register(rc, parents);

        self.steps.insert(node_id, step);
        self.actions_builder.insert(node_id, actions);
    }

    pub fn backward<NC: NodeCleaner, B: Backend>(
        &mut self,
        root_node: NodeRef,
        root_tensor: FloatTensor<B>,
        node_id: NodeId,
    ) -> Gradients {
        let step = self.steps.remove(&node_id).expect(
            "Node should have a step registered, did you forget to call \
             `Tensor::register_grad` on the tensor where you need gradients?",
        );
        let builder = self.actions_builder.remove(&node_id).unwrap();

        let mut consumed = Vec::new();
        let tape_result = self.build_tape(node_id, step, builder, &mut consumed);

        if let Some(sync_client) = get_gradient_sync_client::<B>() {
            sync_client.register_device(tape_result.n_required_map, tape_result.sharded_params);
        };

        let grads = Gradients::new::<B>(root_node.clone(), root_tensor);
        let gradients = Self::execute_steps(tape_result.tape, grads, tape_result.checkpointer);

        if let Some(sync_client) = get_gradient_sync_client::<B>() {
            sync_client.wait_gradients_sync();
        };

        // Cleanup
        let mut cleaner = NC::init();
        self.memory_management
            .free_unavailable_nodes(|node_id: &NodeId| {
                self.steps.remove(node_id);
                self.actions_builder.remove(node_id);
                NC::clean(&mut cleaner, node_id);
            });
        for node_id in consumed {
            cleaner.clean(&node_id)
        }

        gradients
    }

    pub(crate) fn free_unused_roots(&mut self, mut on_free_graph: impl FnMut(&NodeId)) {
        self.memory_management.free_unused_roots(|node_id| {
            self.steps.remove(node_id);
            self.actions_builder.remove(node_id);
            on_free_graph(node_id);
        });
    }

    fn build_tape(
        &mut self,
        node: NodeId,
        node_step: StepBoxed,
        mut builder: CheckpointerBuilder,
        consumed: &mut Vec<NodeId>,
    ) -> TapeResult {
        let mut tape = (0..node_step.depth() + 1)
            .map(|_| Vec::with_capacity(1))
            .collect::<Vec<_>>();

        let mut tree = HashMap::default();
        let mut n_required_map = HashMap::<NodeId, usize>::default();
        let mut sharded_params = HashMap::<NodeId, ShardedParams>::default();

        BreadthFirstSearch.traverse(node, node_step, &mut self.steps, |id, step| {
            self.memory_management.consume_node(id);
            // Clean up consumed node
            consumed.push(id);

            let depth = step.depth();
            step.sharded_params()
                .and_then(|params| sharded_params.insert(id, params));

            if let Some(steps) = tape.get_mut(depth) {
                let parents = step
                    .parents()
                    .iter()
                    .map(|p| {
                        match n_required_map.remove(&p.id) {
                            Some(n) => {
                                n_required_map.insert(p.id, n + 1);
                            }
                            None => {
                                n_required_map.insert(p.id, 1);
                            }
                        };
                        p.id
                    })
                    .filter(|s| *s != id);
                tree.insert(id, parents.collect());
                steps.push(step);
            }

            if let Some(node_builder) = self.actions_builder.remove(&id) {
                builder.extend(node_builder);
            }
        });

        let checkpointer = builder.build(NodeTree::new(tree));

        TapeResult {
            tape,
            checkpointer,
            n_required_map,
            sharded_params,
        }
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

        // For checkpointing tests
        #[cfg(feature = "export_tests")]
        assert!(checkpointer.is_empty());

        grads
    }

    pub(crate) fn maybe_useful(&self) -> bool {
        self.memory_management.maybe_useful()
    }
}
