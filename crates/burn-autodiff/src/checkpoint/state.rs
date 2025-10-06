use core::any::Any;

use crate::collections::HashMap;
use crate::graph::NodeId;
use alloc::boxed::Box;

/// In order to accept arbitrary node output in the same hashmap, we need to upcast them to any.
pub(crate) type StateContent = Box<dyn Any + Send>;

#[derive(Debug)]
/// The state contained at one node. Encapsulates the node output if precomputed,
/// or clearly asks that it needs to be recomputed from the parents.
/// Also keeps track of the number of times the state is required so it can be removed
/// from the map of states on its last use.
pub(crate) enum State {
    /// The state was not checkpointed, will need to recompute it from the node's parents
    Recompute { n_required: usize },
    /// The state was checkpointed or computed during retropropagation and can be directly accessed
    Computed {
        state_content: StateContent,
        n_required: usize,
    },
}

impl State {
    /// Returns a reference to the (not yet) downcasted node output, if checkpointed
    pub(crate) fn to_state_content(&self) -> &StateContent {
        match self {
            State::Recompute { n_required: _ } => {
                unreachable!(
                    "Can't get state content of recompute state. A child has likely been accessed before its parents."
                )
            }
            State::Computed {
                state_content,
                n_required: _,
            } => state_content,
        }
    }

    /// Returns a (not yet) downcasted node output, if checkpointed
    pub(crate) fn into_state_content(self) -> StateContent {
        match self {
            State::Recompute { n_required: _ } => {
                unreachable!(
                    "Can't get state content of recompute state. A child has likely been accessed before its parents."
                )
            }
            State::Computed {
                state_content,
                n_required: _,
            } => state_content,
        }
    }

    /// Returns the number of time the state is required
    pub(crate) fn n_required(&self) -> usize {
        match self {
            State::Recompute { n_required } => *n_required,
            State::Computed {
                state_content: _,
                n_required,
            } => *n_required,
        }
    }
}

#[derive(new, Default, Debug)]
/// Links [NodeId]s to their current state
pub struct BackwardStates {
    map: HashMap<NodeId, State>,
}

impl BackwardStates {
    /// Returns the output in the state of the given [NodeId],
    /// and decrements the number of times this state is required.
    /// This function always gives ownership of the output, but will clone it if needed for further uses.
    pub fn get_state<T>(&mut self, node_id: &NodeId) -> T
    where
        T: Clone + Send + 'static,
    {
        // Fetch the state and decrement its number of required
        let state = self.map.remove(node_id).unwrap();
        let remaining_n_required = state.n_required() - 1;

        // Downcast the state to whatever it is supposed to be
        // If still needed after giving ownership, we copy it back to the hashmap
        if remaining_n_required > 0 {
            let new_stored_state = match state {
                State::Recompute { n_required: _ } => unreachable!(),
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

            self.insert_state(*node_id, new_stored_state);

            downcasted
        } else {
            let downcasted = state.into_state_content().downcast::<T>().unwrap();
            *downcasted
        }
    }

    /// Returns a reference to the [State] of the given node
    /// Useful when we need [State] information without needing the underlying tensor
    pub(crate) fn get_state_ref(&self, node_id: &NodeId) -> Option<&State> {
        self.map.get(node_id)
    }

    /// Associates a [State] to its [NodeId]
    pub(crate) fn insert_state(&mut self, node_id: NodeId, state: State) {
        self.map.insert(node_id, state);
    }

    /// Saves the output to the state of the given [NodeId].
    pub fn save<T>(&mut self, node_id: NodeId, saved_output: T)
    where
        T: Clone + Send + 'static,
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

    #[cfg(feature = "export_tests")]
    pub(crate) fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}
