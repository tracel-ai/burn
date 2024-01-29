use std::{any::Any, collections::HashMap, marker::PhantomData};

use burn_tensor::{backend::Backend, Tensor};

use super::{NodeID, NodeRef};

trait RetroForward {
    fn forward(&self, states: &mut HashMap<NodeID, State>);
}

pub struct RetroDiv<B, const D: usize> {
    lhs: NodeID,
    rhs: NodeID,
    out: NodeID,
    _backend: PhantomData<B>,
}

impl<B: Backend, const D: usize> RetroForward for RetroDiv<B, D> {
    fn forward(&self, states: &mut HashMap<NodeID, State>) {
        // We assume hashmap filled with parents
        let lhs: B::FloatTensorPrimitive<D> = *states
            .get(&self.lhs)
            .unwrap()
            .get_state_content()
            .downcast_ref::<B::FloatTensorPrimitive<D>>()
            .unwrap();

        let rhs: B::FloatTensorPrimitive<D> = *states
            .get(&self.rhs)
            .unwrap()
            .get_state_content()
            .downcast_ref::<B::FloatTensorPrimitive<D>>()
            .unwrap();

        let out: Tensor<B, D> = Tensor::<B, D>::from_primitive(B::float_div(lhs, rhs));

        // insert will erase the old Lazy. might be a problem, not sure yet
        states.insert(
            self.out,
            State::Computed {
                state_content: Box::new(out),
                n_required: 1, // TODO arbitrary for now
            },
        );
    }
}

type StateContent = Box<dyn Any + Send + Sync>;

enum State {
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
    fn get_state_content(&self) -> StateContent {
        match self {
            State::Lazy {
                node_id,
                n_required,
            } => unreachable!("A child has been called before its parents"),
            State::Computed {
                state_content,
                n_required,
            } => *state_content,
        }
    }
}

struct States {
    states: HashMap<NodeID, State>,
    retro_forwards: HashMap<NodeID, Box<dyn RetroForward>>,
    nodes: HashMap<NodeID, NodeRef>,
}

impl States {
    pub fn get<B: Backend, const D: usize>(&mut self, node_id: NodeID) -> StateContent {
        // get is called by backward, knowing exactly what it wants, so knows B and D
        // but must not be called recursively as D may change (and it's just a bad idea in general with muts)

        // we should build a topological sort as we only need to make sure parents are called before their children
        // then it's just a matter of adding to the hashmap in an order that won't panic, not caring about types :)
        let node_order: Vec<NodeID> = self.topological_sort(node_id);

        for node in node_order {
            let retro_forward = self.retro_forwards.get(&node).unwrap();
            retro_forward.forward(&mut self.states);
        }

        self.states.get(&node_id).unwrap().get_state_content()
    }

    fn topological_sort(&self, node_id: NodeID) -> Vec<NodeID> {
        // maybe something to do with n_required here
        match self.states.get(&node_id).unwrap() {
            State::Lazy {
                node_id: _,
                n_required: _,
            } => {
                let mut sorted = Vec::new();
                for parent_node in self.parents(node_id.clone()) {
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

    fn parents(&self, node_id: NodeID) -> Vec<NodeID> {
        self.nodes.get(&node_id).unwrap().parents
    }
}

// progressive transformation:
// all operations saved a custom state for their backward (if necessary)
// this does not mean they are able to perform their forward
// we must leave them the ability of computing their backward from their state
// but not assume it means they have what they need for the retro forward
// conclusion: we do not withdraw the original state, but it's useless for checkpointing, therefore by default all operations are lazy (memory bound)
// some ops will do their own backward with their state, some will ask their ancestors to retroforward, but there's no link.
// have performance tests to decide if we keep both or uniformize

// during a backward,
// an operation may need its original inputs
// it will ask the states struct's get. twice if it has two inputs
// OR just once, asking for its own node id, should work just as well

// Not sure where this goes, if anywhere
// let mut node = None;
// match self.states.get(&node_id).unwrap() {
//     State::Computed { state, n_required } => {
//         // + logic of n_required?
//         return *state;
//     }
//     State::Lazy {
//         node_id,
//         n_required,
//     } => {
//         // + logic of n_required?
//         node = Some(node_id);
//     }
// }
