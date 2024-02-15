use std::{any::Any, collections::HashMap, sync::Arc};

use crate::{
    graph::{NodeID, NodeRef, NodeSteps},
    ops::CheckpointingAction,
};
use std::fmt::Debug;

use super::state::{BackwardStates, State};

/// Definition of the forward function of a node, called during retropropagation only.
/// This is different from the normal forward function because it reads and writes from
/// the [InnerStates] map instead of having a clear function signature.
pub trait RetroForward: Debug + Send + Sync + 'static {
    fn forward(&self, states: &mut BackwardStates, out_node: NodeID);
}

#[derive(new, Default, Debug)]
/// Links [NodeID]s to their corresponding [RetroForward]
pub(crate) struct RetroForwards {
    map: HashMap<NodeID, Arc<dyn RetroForward>>,
}

impl RetroForwards {
    /// Executes the [RetroForward] for a given [NodeID] if the node's
    /// [State] is [State::Recompute], otherwise does nothing.
    fn execute_retro_forward(&mut self, node_id: NodeID, backward_states: &mut BackwardStates) {
        let n_required = match backward_states
            .get_state_ref(&node_id)
            .expect(&format!("Should find node {:?}", node_id))
        {
            State::Recompute { n_required } => *n_required,
            State::Computed {
                state_content: _,
                n_required: _,
            } => return,
        };

        let retro_forward = self.map.remove(&node_id).unwrap();
        retro_forward.forward(backward_states, node_id.clone());
        if n_required > 1 {
            self.map.insert(node_id, retro_forward);
        }
    }

    /// Associates a [RetroForward] to its [NodeID]
    pub(crate) fn insert_retro_forward(
        &mut self,
        node_id: NodeID,
        retro_forward: Arc<dyn RetroForward>,
    ) {
        self.map.insert(node_id, retro_forward);
    }

    pub(crate) fn extend(&mut self, other: Self) {
        self.map.extend(other.map);
    }

    pub(crate) fn len(&self) -> usize {
        self.map.len()
    }
}

#[derive(new, Default, Debug)]
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

    pub(crate) fn extend(&mut self, other: Self) {
        self.map.extend(other.map);
    }

    pub(crate) fn print(&self) {
        println!("{:?}", self.map.keys())
    }
}

#[derive(new, Debug, Default)]
/// Struct responsible of fetching the output for a node in the autodiff graph during a backward pass
pub struct Checkpointer {
    backward_states: BackwardStates,
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
        println!("retrieve {:?}", node_id);
        self.topological_sort(node_id.clone())
            .into_iter()
            .for_each(|node| {
                self.retro_forwards
                    .execute_retro_forward(node, &mut self.backward_states)
            });

        self.backward_states.get_state::<T>(&node_id)
    }
    /// Sorts the ancestors of NodeID in a way such that all parents come before their children
    /// Useful to avoid recursivity later when mutating the states
    fn topological_sort(&self, node_id: NodeID) -> Vec<NodeID> {
        match self.backward_states.get_state_ref(&node_id) {
            Some(state) => match state {
                State::Recompute { n_required: _ } => {
                    let mut sorted = Vec::new();
                    for parent_node in self.node_tree.parents(&node_id) {
                        // if !sorted.contains(&parent_node) {
                        //     sorted.extend(self.topological_sort(parent_node));
                        // }
                        let parent_sorted = self.topological_sort(parent_node);
                        for ps in parent_sorted {
                            if !sorted.contains(&ps) {
                                sorted.push(ps)
                            }
                        }
                    }
                    sorted.push(node_id);
                    println!("X {:?}", sorted);
                    sorted
                }
                State::Computed {
                    state_content: _,
                    n_required: _,
                } => vec![node_id],
            },
            None => panic!("Node {:?} is not in the backward_states. ", node_id),
        }
    }

    pub fn extend(&mut self, other: Self) {
        self.backward_states.extend(other.backward_states);
        self.node_tree.extend(other.node_tree);
        self.retro_forwards.extend(other.retro_forwards);
    }

    pub fn len(&self) -> usize {
        self.backward_states.len() + self.retro_forwards.len()
    }

    /// Insert a [State::Precomputed] at [NodeID]
    fn checkpoint_compute(
        &mut self,
        node_id: NodeID,
        state_content: Box<dyn Any + Send + Sync>,
        n_required: usize,
    ) {
        self.backward_states.insert_state(
            node_id,
            State::Computed {
                state_content,
                n_required,
            },
        );
    }

    fn checkpoint_lazy(
        &mut self,
        node_id: NodeID,
        retro_forward: Arc<dyn RetroForward>,
        n_required: usize,
    ) {
        self.retro_forwards
            .insert_retro_forward(node_id.clone(), retro_forward);
        self.backward_states
            .insert_state(node_id.clone(), State::Recompute { n_required });
    }

    // // TODO TMP
    pub fn print(&self) {
        println!("\n\nCheckpointer");
        println!("\n");
        self.node_tree.print();
        println!("\n{:?}", self.backward_states);
        println!("\n{:?}", self.retro_forwards);

        println!("\n\n");
    }

    pub fn build(
        mut checkpointing_actions: Vec<CheckpointingAction>,
        mut unsure_checkpointing_actions: Vec<CheckpointingAction>,
        graph: &NodeSteps,
    ) -> Self {
        let mut checkpointer = Self::default();
        checkpointer.make_tree(graph);
        let mut n_required_map = HashMap::<NodeID, usize>::default();

        // First loop: computes n_required
        for action in checkpointing_actions.iter() {
            match action {
                CheckpointingAction::Compute {
                    node_ref,
                    state_content: _,
                } => {
                    let id = node_ref.id.clone();
                    println!("{:?}C", id);
                    match n_required_map.remove(&id) {
                        Some(n) => {
                            n_required_map.insert(id, n + 1);
                        }
                        None => {
                            n_required_map.insert(id, 1);
                        }
                    };
                }
                CheckpointingAction::Recompute {
                    node_ref,
                    retro_forward: _,
                } => {
                    let id = node_ref.id.clone();
                    println!("{:?}R", id);
                    checkpointer.find_n_required_of_parents(id, &mut n_required_map);
                }
            }
        }
        println!("{:?}", n_required_map);

        // Second loop: inserts the states into the checkpointer
        // We do not loop over checkpointing actions anymore because they can contain
        // duplicates or miss unsure ones
        // but how to distinguish compute from lazy?
        for (node_id, n_required) in n_required_map {
            // Find it in checkpointing_actions
            // Technically can be there several times but can never be of both types, so we can assume any one is fine
            // Performance could be upgraded by saving index in the loop above, but probably marginal
            let mut pos = None;
            let mut in_unsure = false;
            for (i, action) in checkpointing_actions.iter().enumerate() {
                if action.id() == node_id {
                    pos = Some(i);
                    break;
                }
            }

            if pos.is_none() {
                in_unsure = true;
                for (i, action) in unsure_checkpointing_actions.iter().enumerate() {
                    if action.id() == node_id {
                        pos = Some(i);
                        break;
                    }
                }
            }

            let pos = pos.expect(&format!(
                "Node {:?} is needed but never checkpointed",
                &node_id
            ));

            let action = if in_unsure {
                unsure_checkpointing_actions.remove(pos)
            } else {
                checkpointing_actions.remove(pos)
            };

            match action {
                CheckpointingAction::Compute {
                    node_ref: _,
                    state_content,
                } => checkpointer.checkpoint_compute(node_id, state_content, n_required),
                CheckpointingAction::Recompute {
                    node_ref: _,
                    retro_forward,
                } => checkpointer.checkpoint_lazy(node_id, retro_forward, n_required),
            };
        }

        checkpointer
    }

    fn make_tree(&mut self, graph: &NodeSteps) {
        for (id, step) in graph {
            self.node_tree.insert_node(id.clone(), step.node());
        }
    }

    fn find_n_required_of_parents(&self, id: NodeID, n_required_map: &mut HashMap<NodeID, usize>) {
        println!("{:?}", id);
        match n_required_map.remove(&id) {
            Some(n) => {
                n_required_map.insert(id, n + 1);
            }
            None => {
                let parents = self.node_tree.parents(&id);
                n_required_map.insert(id, 1);
                for p in parents {
                    self.find_n_required_of_parents(p, n_required_map);
                }
            }
        }
    }
}

// new edge case
// a*a -> i think it will be two n_required but consumed once only?
// while a(b*a) is fine because a is in separate nodes?
