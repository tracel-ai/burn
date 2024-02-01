use std::collections::HashMap;

use burn_tensor::{backend::Backend, Tensor};

use crate::graph::{NodeID, NodeRef};

use super::{retro::RetroForward, state::State};

#[derive(new, Default)]
pub(crate) struct RetroForwards {
    map: HashMap<NodeID, Box<dyn RetroForward>>,
}

impl RetroForwards {
    pub fn forward(&self, node_id: &NodeID, inner_states: &mut InnerStates) {
        // elsewhere because needs <B,D>. here we'll assume it's lazy and then B,D can be unknown
        if let State::Lazy {
            node_id,
            n_required: _,
        } = inner_states.get_ref(node_id)
        {
            self.map.get(&node_id).unwrap().forward(inner_states);
        }
    }

    pub fn insert(&mut self, node_id: NodeID, retro_forward: Box<dyn RetroForward>) {
        self.map.insert(node_id, retro_forward);
    }
}

// wrapper to keep track of n_requiered. if zero remove and give ownership. in the REAL forward +=1 n_required
#[derive(new, Default)]
pub(crate) struct InnerStates {
    // We wrap inside an arc because when we get we might or might not have ownership.
    // Arc allows to track that. It's not for tracking n_required.
    map: HashMap<NodeID, State>,
}

impl InnerStates {
    pub fn get_own<B: Backend, const D: usize>(&mut self, node_id: &NodeID) -> Tensor<B, D> {
        let state = self.map.remove(node_id).unwrap();
        let n_required = state.n_required();

        let tensor = state
            .get_state_content()
            .downcast_ref::<Tensor<B, D>>()
            .unwrap()
            .clone();

        // decrement and clone
        let new_stored_state = match state {
            State::Lazy {
                node_id,
                n_required,
            } => State::Lazy {
                node_id,
                n_required: n_required - 1,
            },
            State::Computed {
                state_content,
                n_required,
            } => State::Computed {
                state_content: Box::new(tensor.clone()),
                n_required: n_required - 1,
            },
        };

        if n_required > 0 {
            self.insert(node_id.clone(), new_stored_state);
        }

        tensor
    }

    pub fn get_ref(&self, node_id: &NodeID) -> &State {
        // useful when don't know B, D
        self.map.get(node_id).unwrap()
    }

    pub fn insert(&mut self, node_id: NodeID, state: State) {
        self.map.insert(node_id, state);
    }
}

#[derive(new, Default)]
pub(crate) struct NodeTree {
    map: HashMap<NodeID, NodeRef>,
}

impl NodeTree {
    pub fn parents(&self, node_id: &NodeID) -> Vec<NodeID> {
        self.map.get(node_id).unwrap().parents.clone()
    }

    pub fn insert(&mut self, node_id: NodeID, node_ref: NodeRef) {
        self.map.insert(node_id, node_ref);
    }
}

#[derive(new)]
pub(crate) struct Checkpoint {
    inner_states: InnerStates,
    retro_forwards: RetroForwards,
    node_tree: NodeTree,
}

impl Checkpoint {
    pub fn get<B: Backend, const D: usize>(&mut self, node_id: NodeID) -> Tensor<B, D> {
        self.topological_sort(node_id.clone())
            .iter()
            .for_each(|node| self.retro_forwards.forward(&node, &mut self.inner_states));

        self.inner_states.get_own::<B, D>(&node_id)
    }

    pub fn insert_pre_computed(&mut self, node_id: NodeID, state: State) {
        if let State::Computed {
            state_content: _,
            n_required: _,
        } = state
        {
            self.inner_states.insert(node_id, state);
        } else {
            panic!("Can't insert Lazy state manually")
        }
    }

    fn topological_sort(&self, node_id: NodeID) -> Vec<NodeID> {
        match self.inner_states.get_ref(&node_id) {
            State::Lazy {
                node_id: _,
                n_required: _,
            } => {
                let mut sorted = Vec::new();
                for parent_node in self.node_tree.parents(&node_id) {
                    sorted.extend(self.topological_sort(parent_node));
                }
                sorted.push(node_id);
                sorted
            }
            State::Computed {
                state_content: _,
                n_required: _,
            } => vec![node_id],
        }
    }
}
