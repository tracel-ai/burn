use std::{any::Any, collections::HashMap, sync::Arc};

use crate::{
    graph::{NodeID, NodeSteps},
    ops::CheckpointingAction,
};

use super::{
    base::{Checkpointer, NodeTree, RetroForward, RetroForwards},
    state::{BackwardStates, State},
};

#[derive(new, Debug, Default)]
pub struct CheckpointerBuilder {
    pub main_actions: Vec<CheckpointingAction>,
    pub backup_actions: Vec<CheckpointingAction>,
}

impl CheckpointerBuilder {
    pub(crate) fn extend(&mut self, other: CheckpointerBuilder) {
        for other_action in other.main_actions {
            self.main_actions.push(other_action)
        }
        for other_unsure in other.backup_actions {
            self.backup_actions.push(other_unsure)
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.main_actions.len() + self.backup_actions.len()
    }

    pub(crate) fn build(self, graph: &NodeSteps) -> Checkpointer {
        let node_tree = self.make_tree(graph);
        let mut backward_states_map = HashMap::new();
        let mut retro_forwards_map = HashMap::new();

        // Find recursion stopping points
        let stop_nodes: Vec<NodeID> = self.find_stop_nodes();

        // We start by identifying how many times each node will be required.
        let n_required_map = self.build_n_required_map(&node_tree, stop_nodes);

        // Then we checkpoint the nodes with the corresponding n_required value
        self.insert_checkpoints(
            &mut backward_states_map,
            &mut retro_forwards_map,
            n_required_map,
        );

        Checkpointer::new(
            BackwardStates::new(backward_states_map),
            RetroForwards::new(retro_forwards_map),
            node_tree,
        )
    }

    fn find_stop_nodes(&self) -> Vec<NodeID> {
        let mut stop_nodes = Vec::default();
        for action in self.main_actions.iter().chain(self.backup_actions.iter()) {
            match action {
                CheckpointingAction::Computed {
                    node_ref,
                    state_content: _,
                } => stop_nodes.push(node_ref.id.clone()),
                CheckpointingAction::Recompute {
                    node_ref: _,
                    retro_forward: _,
                } => {}
            }
        }
        stop_nodes
    }

    fn build_n_required_map(
        &self,
        node_tree: &NodeTree,
        stop_nodes: Vec<NodeID>,
    ) -> HashMap<NodeID, usize> {
        let mut n_required_map = HashMap::<NodeID, usize>::default();

        for action in self.main_actions.iter() {
            match action {
                CheckpointingAction::Computed {
                    node_ref,
                    state_content: _,
                } => {
                    let id = node_ref.id.clone();
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
                    self.find_n_required_of_parents(
                        id,
                        &mut n_required_map,
                        node_tree,
                        &stop_nodes,
                    );
                }
            }
        }

        n_required_map
    }

    fn insert_checkpoints(
        mut self,
        backward_states_map: &mut HashMap<NodeID, State>,
        retro_forward_map: &mut HashMap<NodeID, Arc<dyn RetroForward>>,
        n_required_map: HashMap<NodeID, usize>,
    ) {
        // We do not loop over checkpointing actions anymore because they can contain
        // duplicates or miss some that are in backup
        for (node_id, n_required) in n_required_map {
            // We find the checkpointing action for node_id. It's likely in checkpointing_actions
            // so we check there first, otherwise it will be in backup.
            // Technically it can be there several times but can never be of both types, so we can assume the first we find is fine

            let action = match self
                .main_actions
                .iter()
                .position(|action| action.id() == node_id)
            {
                Some(pos) => self.main_actions.remove(pos),
                None => {
                    let pos = self
                        .backup_actions
                        .iter()
                        .position(|action| action.id() == node_id);
                    self.backup_actions.remove(pos.unwrap_or_else(|| {
                        panic!("Node {:?} is needed but never checkpointed", &node_id)
                    }))
                }
            };

            match action {
                CheckpointingAction::Computed {
                    node_ref: _,
                    state_content,
                } => {
                    self.checkpoint_compute(backward_states_map, node_id, state_content, n_required)
                }
                CheckpointingAction::Recompute {
                    node_ref: _,
                    retro_forward,
                } => self.checkpoint_lazy(
                    backward_states_map,
                    retro_forward_map,
                    node_id,
                    retro_forward,
                    n_required,
                ),
            };
        }
    }

    fn make_tree(&self, graph: &NodeSteps) -> NodeTree {
        let mut tree = HashMap::default();
        for (id, step) in graph {
            tree.insert(id.clone(), step.node());
        }
        NodeTree::new(tree)
    }

    fn find_n_required_of_parents(
        &self,
        id: NodeID,
        n_required_map: &mut HashMap<NodeID, usize>,
        node_tree: &NodeTree,
        stop_nodes: &Vec<NodeID>,
    ) {
        match n_required_map.remove(&id) {
            Some(n) => {
                n_required_map.insert(id, n + 1);
            }
            None => {
                n_required_map.insert(id.clone(), 1);
                if !stop_nodes.contains(&id) {
                    if let Some(parents) = node_tree.parents(&id) {
                        for p in parents {
                            self.find_n_required_of_parents(
                                p,
                                n_required_map,
                                node_tree,
                                stop_nodes,
                            );
                        }
                    }
                }
            }
        }
    }

    fn checkpoint_compute(
        &self,
        backward_states_map: &mut HashMap<NodeID, State>,
        node_id: NodeID,
        state_content: Box<dyn Any + Send + Sync>,
        n_required: usize,
    ) {
        backward_states_map.insert(
            node_id,
            State::Computed {
                state_content,
                n_required,
            },
        );
    }

    fn checkpoint_lazy(
        &self,
        backward_states_map: &mut HashMap<NodeID, State>,
        retro_forward_map: &mut HashMap<NodeID, Arc<dyn RetroForward>>,
        node_id: NodeID,
        retro_forward: Arc<dyn RetroForward>,
        n_required: usize,
    ) {
        retro_forward_map.insert(node_id.clone(), retro_forward);
        backward_states_map.insert(node_id.clone(), State::Recompute { n_required });
    }
}
