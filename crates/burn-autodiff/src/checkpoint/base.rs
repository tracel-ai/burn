use super::{
    retro_forward::RetroForwards,
    state::{BackwardStates, State},
};
use crate::collections::HashMap;
use crate::graph::NodeId;

use alloc::{vec, vec::Vec};

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
        self.topological_sort(node_id).into_iter().for_each(|node| {
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
    fn topological_sort(&self, node_id: NodeId) -> Vec<NodeId> {
        match self.backward_states.get_state_ref(&node_id) {
            Some(state) => match state {
                State::Recompute { n_required: _ } => {
                    let mut sorted = Vec::new();
                    let parents = self.node_tree.parents(&node_id).unwrap();
                    for parent_node in parents {
                        let parent_sorted = self.topological_sort(parent_node);
                        for ps in parent_sorted {
                            if !sorted.contains(&ps) {
                                sorted.push(ps)
                            }
                        }
                    }
                    sorted.push(node_id);
                    sorted
                }
                State::Computed {
                    state_content: _,
                    n_required: _,
                } => vec![node_id],
            },
            None => panic!("Node {node_id:?} is not in the backward_states. "),
        }
    }

    #[cfg(feature = "export_tests")]
    /// Checks if checkpointer has been drained adequately. Useful for testing
    pub fn is_empty(&self) -> bool {
        self.backward_states.is_empty() && self.retro_forwards.is_empty()
    }
}
