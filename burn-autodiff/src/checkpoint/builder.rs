use std::{any::Any, collections::HashMap, sync::Arc};

use crate::{
    graph::{CheckpointingActions, NodeID, NodeSteps},
    ops::CheckpointingAction,
};

use super::{
    base::{Checkpointer, NodeTree, RetroForward, RetroForwards},
    state::{BackwardStates, State},
};

// B
pub fn build_checkpointer(
    checkpointing_actions: CheckpointingActions,
    graph: &NodeSteps,
) -> Checkpointer {
    let node_tree = make_tree(graph);
    let mut backward_states_map = HashMap::new();
    let mut retro_forwards_map = HashMap::new();

    // Split checkpointing actions into its two inner vecs
    let main_actions = checkpointing_actions.main_actions;
    let backup_actions = checkpointing_actions.backup_actions;

    // We start by identifying how many times each node will be required.
    let n_required_map = build_n_required_map(&main_actions, &node_tree);

    // Then we checkpoint the nodes with the corresponding n_required value
    insert_checkpoints(
        &mut backward_states_map,
        &mut retro_forwards_map,
        n_required_map,
        main_actions,
        backup_actions,
    );

    println!("Checkpointer successfully built");
    println!("{:?}", backward_states_map);
    Checkpointer::new(
        BackwardStates::new(backward_states_map),
        RetroForwards::new(retro_forwards_map),
        node_tree,
    )
}

fn build_n_required_map(
    checkpointing_actions: &Vec<CheckpointingAction>,
    node_tree: &NodeTree,
) -> HashMap<NodeID, usize> {
    let mut n_required_map = HashMap::<NodeID, usize>::default();

    for action in checkpointing_actions.iter() {
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
                find_n_required_of_parents(id, &mut n_required_map, node_tree);
            }
        }
    }

    println!("{:?}", n_required_map);
    n_required_map
}

fn insert_checkpoints(
    backward_states_map: &mut HashMap<NodeID, State>,
    retro_forward_map: &mut HashMap<NodeID, Arc<dyn RetroForward>>,
    n_required_map: HashMap<NodeID, usize>,
    mut checkpointing_actions: Vec<CheckpointingAction>,
    mut backup_checkpointing_actions: Vec<CheckpointingAction>,
) {
    println!("{:?}", checkpointing_actions);
    println!("{:?}", backup_checkpointing_actions);
    // We do not loop over checkpointing actions anymore because they can contain
    // duplicates or miss some that are in backup
    for (node_id, n_required) in n_required_map {
        // We find the checkpointing action for node_id. It's likely in checkpointing_actions
        // so we check there first, otherwise it will be in backup.
        // Technically it can be there several times but can never be of both types, so we can assume the first we find is fine

        let action = match checkpointing_actions
            .iter()
            .position(|action| action.id() == node_id)
        {
            Some(pos) => checkpointing_actions.remove(pos),
            None => {
                let pos = backup_checkpointing_actions
                    .iter()
                    .position(|action| action.id() == node_id);
                backup_checkpointing_actions.remove(pos.expect(&format!(
                    "Node {:?} is needed but never checkpointed",
                    &node_id
                )))
            }
        };

        match action {
            CheckpointingAction::Computed {
                node_ref: _,
                state_content,
            } => checkpoint_compute(backward_states_map, node_id, state_content, n_required),
            CheckpointingAction::Recompute {
                node_ref: _,
                retro_forward,
            } => checkpoint_lazy(
                backward_states_map,
                retro_forward_map,
                node_id,
                retro_forward,
                n_required,
            ),
        };
    }
}

fn make_tree(graph: &NodeSteps) -> NodeTree {
    let mut tree = HashMap::default();
    for (id, step) in graph {
        println!("{:?}", id);
        tree.insert(id.clone(), step.node());
    }
    NodeTree::new(tree)
}

fn find_n_required_of_parents(
    id: NodeID,
    n_required_map: &mut HashMap<NodeID, usize>,
    node_tree: &NodeTree,
) {
    match n_required_map.remove(&id) {
        Some(n) => {
            n_required_map.insert(id, n + 1);
        }
        None => {
            if let Some(parents) = node_tree.parents(&id) {
                n_required_map.insert(id, 1);
                for p in parents {
                    find_n_required_of_parents(p, n_required_map, node_tree);
                }
            }
        }
    }
}

fn checkpoint_compute(
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
    backward_states_map: &mut HashMap<NodeID, State>,
    retro_forward_map: &mut HashMap<NodeID, Arc<dyn RetroForward>>,
    node_id: NodeID,
    retro_forward: Arc<dyn RetroForward>,
    n_required: usize,
) {
    retro_forward_map.insert(node_id.clone(), retro_forward);
    backward_states_map.insert(node_id.clone(), State::Recompute { n_required });
}
