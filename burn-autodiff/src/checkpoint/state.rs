use std::any::Any;

/// In order to accept arbitrary node output in the same hashmap, we need to upcast them to any.
pub(crate) type StateContent = Box<dyn Any + Send + Sync>;

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
                unreachable!("A child has been accessed before its parents")
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
                unreachable!("A child has been accessed before its parents")
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
            State::Recompute { n_required } => n_required.clone(),
            State::Computed {
                state_content: _,
                n_required,
            } => n_required.clone(),
        }
    }
}
