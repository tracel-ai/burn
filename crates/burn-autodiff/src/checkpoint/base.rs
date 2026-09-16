use super::{
    retro_forward::RetroForwards,
    state::{BackwardStates, State},
};
use crate::collections::{HashMap, HashSet};
use crate::graph::NodeId;

use alloc::{format, vec::Vec};
use burn_std::config::{autodiff::AutodiffLogLevel, log_autodiff};

#[derive(new, Debug)]
/// Links a [NodeId] to its autodiff graph [NodeRef]
pub(crate) struct NodeTree {
    map: HashMap<NodeId, Vec<NodeId>>,
}

impl NodeTree {
    /// Gives the parents of the node in the autodiff graph
    pub(crate) fn parents(&self, node_id: &NodeId) -> Option<Vec<NodeId>> {
        self.map.get(node_id).cloned()
    }
}

#[derive(new, Debug)]
/// Struct responsible of fetching the output for a node in the autodiff graph during a backward pass
pub struct Checkpointer {
    backward_states: BackwardStates,
    retro_forwards: RetroForwards,
    node_tree: NodeTree,
}

impl Checkpointer {
    /// Gives the output of the given node, by recursively asking parents to compute themselves
    /// or give their pre-computed tensors.
    pub fn retrieve_node_output<T>(&mut self, node_id: NodeId) -> T
    where
        T: Clone + Send + 'static,
    {
        let sorted = self.topological_sort(node_id);
        let num_nodes = sorted.len();
        log_autodiff(AutodiffLogLevel::Basic, move || {
            format!("retrieve_node_output {node_id:?}: {num_nodes} node(s) to compute")
        });

        sorted.into_iter().for_each(|node| {
            log_autodiff(AutodiffLogLevel::Full, move || {
                format!("execute_retro_forward {node:?}")
            });
            self.retro_forwards
                .execute_retro_forward(node, &mut self.backward_states)
        });

        self.backward_states.get_state::<T>(&node_id)
    }

    /// Sorts the ancestors of NodeId in a way such that all parents come before their children
    /// Useful to avoid recursivity later when mutating the states
    ///
    /// The sort on a compute bound state or a memory bound that is already computed is trivial.
    /// The match on State::Computed also serves as a stopping criterion for the sort,
    /// we don't need to look higher than that during recursivity.
    ///
    /// Ancestors shared by multiple branches (e.g. a recurrent chain revisited through several
    /// paths) are only expanded once: a `visited` set is checked before recursing into a node's
    /// parents at all, not just before appending it. Without that, both the membership check and
    /// the re-expansion of already-sorted subtrees are redone on every branch, which turns a
    /// single retrieval into quadratic (or worse) work as the ancestor chain grows.
    fn topological_sort(&self, node_id: NodeId) -> Vec<NodeId> {
        let mut sorted = Vec::new();
        let mut visited = HashSet::new();
        self.topological_sort_visit(node_id, &mut visited, &mut sorted);
        sorted
    }

    fn topological_sort_visit(
        &self,
        node_id: NodeId,
        visited: &mut HashSet<NodeId>,
        sorted: &mut Vec<NodeId>,
    ) {
        if !visited.insert(node_id) {
            return;
        }

        match self.backward_states.get_state_ref(&node_id) {
            Some(state) => match state {
                State::Recompute { n_required: _ } => {
                    let parents = self.node_tree.parents(&node_id).unwrap();
                    for parent_node in parents {
                        self.topological_sort_visit(parent_node, visited, sorted);
                    }
                    sorted.push(node_id);
                }
                State::Computed {
                    state_content: _,
                    n_required: _,
                } => sorted.push(node_id),
            },
            None => panic!("Node {node_id:?} is not in the backward_states. "),
        }
    }

    /// Checks if checkpointer has been drained adequately. Useful for testing
    pub fn is_empty(&self) -> bool {
        self.backward_states.is_empty() && self.retro_forwards.is_empty()
    }
}
