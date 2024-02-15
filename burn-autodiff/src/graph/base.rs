use spin::Mutex;
use std::{collections::HashMap, sync::Arc};

use crate::{checkpoint::base::Checkpointer, grads::Gradients, ops::CheckpointingAction};

use super::{NodeID, NodeRef};

/// Backward step for reverse mode autodiff.
pub trait Step: Send + Sync + std::fmt::Debug {
    /// Executes the step and consumes it.
    fn step(self: Box<Self>, grads: &mut Gradients, checkpointer: &mut Checkpointer);
    /// The node associated to the step.
    fn node(&self) -> NodeRef;
}

pub type StepBoxed = Box<dyn Step>;
pub type NodeSteps = HashMap<NodeID, StepBoxed>;

#[derive(new, Debug, Default)]
pub struct CheckpointingActions {
    pub actions: Vec<CheckpointingAction>,
    pub unsure: Vec<CheckpointingAction>,
}

impl CheckpointingActions {
    fn extend(&mut self, other: CheckpointingActions) {
        for other_action in other.actions {
            self.actions.push(other_action)
        }
        for other_unsure in other.unsure {
            self.unsure.push(other_unsure)
        }
    }

    fn len(&self) -> usize {
        self.actions.len() + self.unsure.len()
    }
}

/// Graph data structure.
///
/// The graph contains the [node steps](Step), which can be access by [node id](NodeID).
#[derive(Default, Clone, Debug)]
pub struct Graph {
    steps: Arc<Mutex<NodeSteps>>,
    checkpointing_actions: Arc<Mutex<CheckpointingActions>>,
}

impl Graph {
    /// Create a new graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get all the steps for the graph.
    ///
    /// # Notes
    ///
    /// This is a owned method, so the current graph will be freed. However, the steps can
    /// be shared with other graphs, therefore they are going to be cleared.
    ///
    /// This is useful, since the graph is supposed to be consumed only once for backprop, and
    /// keeping all the tensors alive for multiple backward call is a heavy waste of resources.
    pub fn steps(self) -> NodeSteps {
        let mut map_drain = HashMap::new();
        self.execute_mut_steps(|map| {
            std::mem::swap(&mut *map, &mut map_drain);
        });
        map_drain
    }

    /// Register a new step into the graph.
    pub fn register(self, id: &NodeID, ops: StepBoxed) -> Self {
        self.execute_mut_steps(|map| {
            map.insert(id.clone(), ops);
        })
    }

    /// Merge two graphs.
    pub fn merge(self, other: Self) -> Self {
        if Arc::ptr_eq(&self.steps, &other.steps) {
            return self;
        }

        self.merge_different(other)
    }

    fn execute_mut_steps<F: FnOnce(&mut NodeSteps)>(mut self, func: F) -> Self {
        match Arc::get_mut(&mut self.steps) {
            Some(mutex) => {
                let map = mutex.get_mut();
                func(map);
            }
            None => {
                // Only lock when there are multiple references to the graph.
                let mut map = self.steps.lock();
                func(&mut map);
            }
        };

        self
    }

    fn merge_different(self, other: Self) -> Self {
        let mut map2 = other.clone().steps();
        let mut actions2 = other.checkpointing_actions_own();

        self.execute_mut_steps(|map1| {
            if map1.len() > map2.len() {
                map1.extend(map2);
            } else {
                let mut map_drain = HashMap::new();
                std::mem::swap(map1, &mut map_drain);
                map2.extend(map_drain);
                std::mem::swap(map1, &mut map2);
            }
        })
        .execute_mut_checkpointing_actions(|actions1| {
            if actions1.len() > actions2.len() {
                actions1.extend(actions2);
            } else {
                let mut checkpointing_drain = CheckpointingActions::default();
                std::mem::swap(actions1, &mut checkpointing_drain);
                actions2.extend(checkpointing_drain);
                std::mem::swap(actions1, &mut actions2);
            }
        })
    }

    /// # Notes
    ///
    /// This is a owned method, so the current checkpointer will be freed.
    pub fn checkpointing_actions_own(self) -> CheckpointingActions {
        let mut actions = CheckpointingActions::default();
        self.execute_mut_checkpointing_actions(|checkpointing_actions| {
            std::mem::swap(&mut *checkpointing_actions, &mut actions);
        });
        actions
    }

    fn execute_mut_checkpointing_actions<F: FnOnce(&mut CheckpointingActions)>(
        mut self,
        func: F,
    ) -> Self {
        match Arc::get_mut(&mut self.checkpointing_actions) {
            Some(mutex) => {
                let map = mutex.get_mut();
                func(map);
            }
            None => {
                // Only lock when there are multiple references to the graph.
                let mut actions = self.checkpointing_actions.lock();
                func(&mut actions);
            }
        };

        self
    }

    // pub fn take_checkpointing_actions(&self) -> Vec<CheckpointingAction> {
    //     let mut guard = self.checkpointing_actions.lock();
    //     let owned: Vec<CheckpointingAction> = std::mem::replace(&mut *guard, Vec::default());
    //     owned
    // }

    pub(crate) fn build_checkpointer(&self) -> Checkpointer {
        let mut guard = self.checkpointing_actions.lock();
        let owned: CheckpointingActions =
            std::mem::replace(&mut *guard, CheckpointingActions::default());
        Checkpointer::build(owned.actions, owned.unsure, &self.steps.lock())
    }

    // pub fn print_checkpoint(&self) {
    //     self.checkpointer.lock().print();
    // }

    // pub(crate) fn merge_checkpointer(self, checkpointer: Checkpointer) -> Self {
    //     // TODO add again the if for slarget one
    //     self.execute_mut_checkpointer(|checkpointer1| {
    //         checkpointer1.extend(checkpointer);
    //     })
    // }

    pub(crate) fn extend_checkpointing_actions(
        &self,
        actions: Vec<CheckpointingAction>,
        unsure: Vec<CheckpointingAction>,
    ) {
        self.checkpointing_actions
            .lock()
            .extend(CheckpointingActions::new(actions, unsure));
    }
}
