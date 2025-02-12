use crate::{
    graph::{ComputingProperty, NodeID, NodeSteps},
    tensor::AutodiffTensor,
};
use burn_tensor::backend::Backend;
use std::{any::Any, collections::HashMap, sync::Arc};

use super::{
    base::{Checkpointer, NodeTree},
    retro_forward::{RetroForward, RetroForwards},
    state::{BackwardStates, State},
};

#[derive(Debug)]
/// Determines if a node should checkpoint its computed output or its retro_forward for recomputation
/// The action is normally created by the child of the node, once the node is determined to be needed
pub enum CheckpointingAction {
    /// The node's already computed output should be saved
    Computed {
        /// The node
        node_id: NodeID,
        /// The node's output
        state_content: Box<dyn Any + Send>,
    },
    /// The node should recompute itself when asked
    Recompute {
        /// The node
        node_id: NodeID,
        /// How the node should recompute itself
        retro_forward: Arc<dyn RetroForward>,
    },
}

// TODO: Remove that when proper client server.
unsafe impl Send for CheckpointingAction {}

impl CheckpointingAction {
    /// Utilitary function to access the id of the node of the checkpointing action
    pub fn id(&self) -> NodeID {
        match self {
            CheckpointingAction::Computed {
                node_id: node_ref,
                state_content: _,
            } => *node_ref,
            CheckpointingAction::Recompute {
                node_id: node_ref,
                retro_forward: _,
            } => *node_ref,
        }
    }
}

#[derive(new, Debug, Default)]
/// Accumulates checkpoints as checkpointing actions during the forward pass,
/// and builds a checkpointer right before the backward pass
pub struct CheckpointerBuilder {
    explicit_actions: Vec<CheckpointingAction>,
    backup_actions: Vec<CheckpointingAction>,
}

/// Determines if a checkpoint should impact the n_required values (Main)
/// or if it should just keep the state in case it's required (Backup)
///
pub(crate) enum ActionType {
    /// Explicit actions have been explicitly requested by some operation to retrieve their state
    Explicit,
    /// Backup actions are not always needed. They exist to save the output of an operation
    /// whose child is memory bound, in case the state is indirectly needed when computing
    /// the child's retro_forward. If no explicit action ever asks for the child's output, then
    /// the backup output will go out of scope when the checkpointer is built.
    Backup,
}

impl CheckpointerBuilder {
    pub(crate) fn checkpoint<B: Backend>(
        &mut self,
        tensor: &AutodiffTensor<B>,
        action_type: ActionType,
    ) {
        let action_list = match action_type {
            ActionType::Explicit => &mut self.explicit_actions,
            ActionType::Backup => &mut self.backup_actions,
        };
        match &tensor.node.properties {
            ComputingProperty::ComputeBound | ComputingProperty::Ambiguous => {
                action_list.push(CheckpointingAction::Computed {
                    node_id: tensor.node.id,
                    state_content: Box::new(tensor.primitive.clone()),
                })
            }
            ComputingProperty::MemoryBound { retro_forward } => {
                action_list.push(CheckpointingAction::Recompute {
                    node_id: tensor.node.id,
                    retro_forward: retro_forward.clone(),
                })
            }
        }
    }

    pub(crate) fn extend(&mut self, other: CheckpointerBuilder) {
        for other_action in other.explicit_actions {
            self.explicit_actions.push(other_action)
        }
        for other_unsure in other.backup_actions {
            self.backup_actions.push(other_unsure)
        }
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
        for action in self
            .explicit_actions
            .iter()
            .chain(self.backup_actions.iter())
        {
            match action {
                CheckpointingAction::Computed {
                    node_id: node_ref,
                    state_content: _,
                } => stop_nodes.push(*node_ref),
                CheckpointingAction::Recompute {
                    node_id: _,
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

        for action in self.explicit_actions.iter() {
            match action {
                CheckpointingAction::Computed {
                    node_id: node_ref,
                    state_content: _,
                } => {
                    let id = *node_ref;
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
                    node_id: node_ref,
                    retro_forward: _,
                } => {
                    let id = *node_ref;
                    Self::update_n_required_of_parents(
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
        // duplicates or miss some that are in backup. We loop over the n_required_map
        // from which we use the ids to find them again in the checkpointing actions
        for (node_id, n_required) in n_required_map {
            // We find the checkpointing action for node_id. It's likely in checkpointing_actions
            // so we check there first, otherwise it will be in backup.
            // Technically it can be there several times but can never be of both types, so we can assume the first we find is fine

            let action = match self
                .explicit_actions
                .iter()
                .position(|action| action.id() == node_id)
            {
                Some(pos) => self.explicit_actions.remove(pos),
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
                    node_id: _,
                    state_content,
                } => {
                    self.checkpoint_compute(backward_states_map, node_id, state_content, n_required)
                }
                CheckpointingAction::Recompute {
                    node_id: _,
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
            tree.insert(*id, step.parents());
        }
        NodeTree::new(tree)
    }

    fn update_n_required_of_parents(
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
                n_required_map.insert(id, 1);
                if !stop_nodes.contains(&id) {
                    if let Some(parents) = node_tree.parents(&id) {
                        for p in parents {
                            Self::update_n_required_of_parents(
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
        state_content: Box<dyn Any + Send>,
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
        retro_forward_map.insert(node_id, retro_forward);
        backward_states_map.insert(node_id, State::Recompute { n_required });
    }
}
