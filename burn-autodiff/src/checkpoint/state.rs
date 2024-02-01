use std::any::Any;

use burn_tensor::{backend::Backend, Tensor};

use crate::graph::NodeID;

pub(crate) type StateContent = Box<dyn Any + Send + Sync>;

#[derive(Debug)]
pub(crate) enum State {
    // Weird nomenclature. Isn't it more lazy to not re-compute?
    Lazy {
        node_id: NodeID, // whose forward is required to compute state (is it needed, as States has it as the key)
        n_required: usize, // how many times it's used (has counter += and -=)
    },
    Computed {
        state_content: StateContent,
        n_required: usize,
    },
}

impl State {
    pub fn get_state_content(&self) -> &StateContent {
        match self {
            State::Lazy {
                node_id: _,
                n_required: _,
            } => unreachable!("A child has been called before its parents"),
            State::Computed {
                state_content,
                n_required: _,
            } => state_content,
        }
    }

    pub fn n_required(&self) -> usize {
        match self {
            State::Lazy {
                node_id,
                n_required,
            } => n_required.clone(),
            Self::Computed {
                state_content,
                n_required,
            } => n_required.clone(),
        }
    }
}
