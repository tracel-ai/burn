use std::collections::HashMap;

use crate::graph::{NodeID, NodeRef};

use super::state::State;

/// Definition of the forward function of a node, called during retropropagation only.
/// This is different from the normal forward function because it reads and writes from
/// the [InnerStates] map instead of having a clear function signature.
pub(crate) trait RetroForward {
    fn forward(&self, states: &mut BackwardStates);
}

#[derive(new, Default)]
/// Links [NodeID]s to their corresponding [RetroForward]
pub(crate) struct RetroForwards {
    map: HashMap<NodeID, Box<dyn RetroForward>>,
}

impl RetroForwards {
    /// Executes the [RetroForward] for a given [NodeID] if the node's
    /// [State] is [State::Recompute], otherwise does nothing.
    fn execute_retro_forward(&mut self, node_id: NodeID, output_states: &mut BackwardStates) {
        let n_required = match output_states.get_state_ref(&node_id).unwrap() {
            State::Recompute { n_required } => n_required.clone(),
            State::Computed {
                state_content: _,
                n_required: _,
            } => return,
        };

        let retro_forward = self.map.remove(&node_id).unwrap();
        retro_forward.forward(output_states);
        if n_required > 1 {
            self.map.insert(node_id, retro_forward);
        }
    }

    /// Associates a [RetroForward] to its [NodeID]
    pub(crate) fn insert_retro_forward(
        &mut self,
        node_id: NodeID,
        retro_forward: Box<dyn RetroForward>,
    ) {
        self.map.insert(node_id, retro_forward);
    }
}

#[derive(new, Default)]
/// Links [NodeID]s to their current [State]
pub(crate) struct BackwardStates {
    map: HashMap<NodeID, State>,
}

impl BackwardStates {
    /// Returns the output in the [State] of the given [NodeID],
    /// and decrements the number of times this state is required.
    /// This function always gives ownership of the output, but will clone it if needed for further uses.
    pub(crate) fn get_state<T>(&mut self, node_id: &NodeID) -> T
    where
        T: Clone + Send + Sync + 'static,
    {
        // Fetch the state and decrement its number of required
        let state = self.map.remove(node_id).unwrap();
        let remaining_n_required = state.n_required() - 1;

        // Downcast the state to whatever it is supposed to be
        // If still needed after giving ownership, we copy it back to the hashmap
        if remaining_n_required > 0 {
            let new_stored_state = match state {
                State::Recompute { n_required: _ } => State::Recompute {
                    n_required: remaining_n_required,
                },
                State::Computed {
                    state_content,
                    n_required: _,
                } => State::Computed {
                    state_content,
                    n_required: remaining_n_required,
                },
            };

            let downcasted = new_stored_state
                .to_state_content()
                .downcast_ref::<T>()
                .unwrap()
                .clone();

            self.insert_state(node_id.clone(), new_stored_state);

            downcasted
        } else {
            let downcasted = state.into_state_content().downcast::<T>().unwrap();
            *downcasted
        }
    }

    /// Returns a reference to the [State] of the given node
    /// Useful when we need [State] information without needing the underlying tensor
    pub(crate) fn get_state_ref(&self, node_id: &NodeID) -> Option<&State> {
        self.map.get(node_id)
    }

    /// Associates a [State] to its [NodeID]
    pub(crate) fn insert_state(&mut self, node_id: NodeID, state: State) {
        self.map.insert(node_id, state);
    }

    pub(crate) fn save<T>(&mut self, node_id: NodeID, saved_output: T)
    where
        T: Clone + Send + Sync + 'static,
    {
        let n_required = self.get_state_ref(&node_id).unwrap().n_required();
        self.insert_state(
            node_id,
            State::Computed {
                state_content: Box::new(saved_output),
                n_required,
            },
        );
    }
}

#[derive(new, Default)]
/// Links a [NodeID] to its autodiff graph [NodeRef]
pub(crate) struct NodeTree {
    map: HashMap<NodeID, NodeRef>,
}

impl NodeTree {
    /// Gives the parents of the node in the autodiff graph
    fn parents(&self, node_id: &NodeID) -> Vec<NodeID> {
        self.map.get(node_id).unwrap().parents.clone()
    }

    // Associates a [NodeRef] to its [NodeID]
    pub(crate) fn insert_node(&mut self, node_id: NodeID, node_ref: NodeRef) {
        self.map.insert(node_id, node_ref);
    }
}

#[derive(new)]
/// Struct responsible of fetching the output for a node in the autodiff graph during a backward pass
pub struct Checkpointer {
    output_states: BackwardStates,
    retro_forwards: RetroForwards,
    node_tree: NodeTree,
}

impl Checkpointer {
    /// Gives the output of the given node, by recursively asking parents to compute themselves
    /// or give their pre-computed tensors.
    pub fn retrieve_output<T>(&mut self, node_id: NodeID) -> T
    where
        T: Clone + Send + Sync + 'static,
    {
        self.topological_sort(node_id.clone())
            .into_iter()
            .for_each(|node| {
                self.retro_forwards
                    .execute_retro_forward(node, &mut self.output_states)
            });

        self.output_states.get_state::<T>(&node_id)
    }

    /// Insert a [State::Precomputed] at [NodeID]
    pub fn checkpoint<T>(&mut self, node_id: NodeID, saved_output: T, n_required: usize)
    where
        T: Clone + Send + Sync + 'static,
    {
        self.output_states.insert_state(
            node_id,
            State::Computed {
                state_content: Box::new(saved_output),
                n_required,
            },
        );
    }

    /// Sorts the ancestors of NodeID in a way such that all parents come before their children
    /// Useful to avoid recursivity later when mutating the states
    fn topological_sort(&self, node_id: NodeID) -> Vec<NodeID> {
        match self.output_states.get_state_ref(&node_id) {
            Some(state) =>
            {
                match state {
                State::Recompute {
                    n_required: _,
                } => {
                    let mut sorted = Vec::new();
                    for parent_node in self.node_tree.parents(&node_id) {
                        if !sorted.contains(&parent_node){
                            sorted.extend(self.topological_sort(parent_node));
                        }
                    }
                    sorted.push(node_id);
                    sorted
                }
                State::Computed {
                    state_content: _,
                    n_required: _,
                } => vec![node_id],
            }}
            None => panic!("Node is not in the map. You may have tried to access it more times than n_required allowed.")
        }
    }
}
