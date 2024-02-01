use std::collections::HashMap;

use crate::graph::{NodeID, NodeRef};

use super::{
    retro::RetroForward,
    state::{State, StateContent},
};

#[derive(new, Default)]
pub(crate) struct RetroForwards {
    map: HashMap<NodeID, Box<dyn RetroForward>>,
}

impl RetroForwards {
    pub fn forward(&self, node_id: &NodeID, inner_states: &mut InnerStates) {
        if let State::Lazy {
            node_id,
            n_required,
        } = inner_states.get(node_id)
        {
            self.map.get(node_id).unwrap().forward(inner_states);
        }
    }

    pub fn insert(&mut self, node_id: NodeID, retro_forward: Box<dyn RetroForward>) {
        self.map.insert(node_id, retro_forward);
    }
}

// wrapper to keep track of n_requiered. if zero remove and give ownership. in the REAL forward +=1 n_required
#[derive(new, Default)]
pub(crate) struct InnerStates {
    map: HashMap<NodeID, State>,
}

impl InnerStates {
    pub fn get(&self, node_id: &NodeID) -> &State {
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
pub(crate) struct States {
    inner_states: InnerStates,
    retro_forwards: RetroForwards,
    node_tree: NodeTree,
}

impl States {
    pub fn get(&mut self, node_id: NodeID) -> &StateContent {
        self.topological_sort(node_id.clone())
            .iter()
            .for_each(|node| self.retro_forwards.forward(&node, &mut self.inner_states));

        self.inner_states.get(&node_id).get_state_content()
    }

    fn topological_sort(&self, node_id: NodeID) -> Vec<NodeID> {
        match self.inner_states.get(&node_id) {
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
