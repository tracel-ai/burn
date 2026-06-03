use super::memory_management::GraphMemoryManagement;
use crate::{
    NodeId,
    checkpoint::{
        base::{Checkpointer, NodeTree},
        builder::CheckpointerBuilder,
    },
    collections::HashMap,
    grads::{BackwardMode, Gradients},
    graph::{
        NodeRef, StepBoxed,
        traversal::{BreadthFirstSearch, TraversalItem},
    },
    tensor::NodeRefCount,
};
use alloc::vec::Vec;
use burn_backend::Backend;
use burn_backend::tensor::FloatTensor;

#[cfg(feature = "distributed")]
use crate::grads::GradSyncContext;

struct TapeResult {
    tape: Vec<Vec<StepBoxed>>,
    checkpointer: Checkpointer,
    #[cfg(feature = "distributed")]
    distributed: Option<GradSyncContext>,
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
        mode: BackwardMode,
    ) -> Gradients {
        let step = self.steps.remove(&node_id).expect(
            "Node should have a step registered, did you forget to call \
             `Tensor::register_grad` on the tensor where you need gradients?",
        );
        let builder = self.actions_builder.remove(&node_id).unwrap();

        let mut consumed = Vec::new();
        let tape_result = self.build_tape(node_id, step, builder, &mut consumed);

        let grads = match mode {
            #[cfg(feature = "distributed")]
            BackwardMode::Distributed(factory) if tape_result.distributed.is_some() => {
                let on_register = factory(tape_result.distributed.clone().unwrap());
                Gradients::new_distributed::<B>(root_node, root_tensor, on_register)
            }
            _ => Gradients::new::<B>(root_node, root_tensor),
        };

        let gradients = Self::execute_steps(tape_result.tape, grads, tape_result.checkpointer);

        self.cleanup::<NC>(&consumed);

        gradients
    }

    /// Execute backward without consuming graph state, allowing repeated backward passes.
    pub fn backward_retain(&mut self, grads: Gradients, node_id: NodeId) -> Gradients {
        let (tape, checkpointer) = self.build_tape_retaining(node_id);
        Self::execute_steps_retaining(tape, &self.steps, grads, checkpointer)
    }

    fn cleanup<NC: NodeCleaner>(&mut self, consumed: &Vec<NodeId>) {
        let mut cleaner = NC::init();
        self.memory_management
            .free_unavailable_nodes(|node_id: &NodeId| {
                self.steps.remove(node_id);
                self.actions_builder.remove(node_id);
                NC::clean(&mut cleaner, node_id);
            });
        for node_id in consumed {
            cleaner.clean(node_id)
        }
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

        #[cfg(feature = "distributed")]
        let mut n_required_map = HashMap::default();
        #[cfg(feature = "distributed")]
        let mut distributed_params = HashMap::default();

        BreadthFirstSearch.traverse(node, node_step, &mut self.steps, |id, step| {
            self.memory_management.consume_node(id);
            // Clean up consumed node
            consumed.push(id);

            let depth = step.depth();

            #[cfg(feature = "distributed")]
            step.distributed_params()
                .and_then(|params| distributed_params.insert(id, params));

            if let Some(steps) = tape.get_mut(depth) {
                let parents = step
                    .parents()
                    .iter()
                    .map(|p| {
                        #[cfg(feature = "distributed")]
                        {
                            *n_required_map.entry(p.id).or_insert(0) += 1;
                        }
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
        #[cfg(feature = "distributed")]
        let distributed = Some(GradSyncContext {
            n_required_map,
            distributed_params,
        });

        TapeResult {
            tape,
            checkpointer,
            #[cfg(feature = "distributed")]
            distributed,
        }
    }

    /// Build tape by borrowing steps and builders, leaving graph state intact.
    fn build_tape_retaining(&mut self, node: NodeId) -> (Vec<Vec<NodeId>>, Checkpointer) {
        let depth = self
            .steps
            .get(&node)
            .expect(
                "Node should have a step registered, did you forget to call \
                 `Tensor::register_grad` on the tensor where you need gradients?",
            )
            .depth();

        let mut tape = (0..depth)
            .map(|_| Vec::with_capacity(1))
            .collect::<Vec<_>>();

        let mut tree = HashMap::default();
        let mut builder = CheckpointerBuilder::default();

        BreadthFirstSearch.traverse_retaining(node, &self.steps, |id, step| {
            let step_depth = step.depth();

            if step_depth == 0 {
                return;
            }

            if let Some(ids) = tape.get_mut(step_depth - 1) {
                let parents = step.parents().iter().map(|p| p.id).filter(|s| *s != id);
                tree.insert(id, parents.collect());
                ids.push(id);
            }

            if let Some(node_builder) = self.actions_builder.get(&id) {
                builder.extend_ref(node_builder);
            }
        });

        let checkpointer = builder.build(NodeTree::new(tree));

        (tape, checkpointer)
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

    fn execute_steps_retaining(
        tape: Vec<Vec<NodeId>>,
        steps: &HashMap<NodeId, StepBoxed>,
        mut grads: Gradients,
        mut checkpointer: Checkpointer,
    ) -> Gradients {
        tape.into_iter().rev().for_each(|ids| {
            ids.into_iter().for_each(|id| {
                let step = steps.get(&id).expect("Step must exist for retained graph");
                step.step(&mut grads, &mut checkpointer);
            })
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
