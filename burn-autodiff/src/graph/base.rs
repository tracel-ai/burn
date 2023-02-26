use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use burn_tensor::backend::Backend;

use crate::grads::Gradients;

use super::{NodeID, NodeRef};

/// Backward step for reverse mode autodiff.
pub trait Step<B: Backend>: Send + Sync + std::fmt::Debug {
    /// Execute the step and consume it.
    fn step(self: Box<Self>, grads: &mut Gradients<B>);
    /// The node associated to the step.
    fn node(&self) -> NodeRef;
}

pub type StepBoxed<B> = Box<dyn Step<B>>;
pub type NodeSteps<B> = HashMap<NodeID, Box<dyn Step<B>>>;

/// Graph data structure.
///
/// The graph contains the [node steps](Step), which can be access by [node id](NodeID).
#[derive(Default, Clone)]
pub struct Graph<B: Backend> {
    steps: Arc<Mutex<NodeSteps<B>>>,
}

impl<B: Backend> std::fmt::Debug for Graph<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(format!("OpsMap<{:?}>", B::name()).as_str())
    }
}

impl<B: Backend> Graph<B> {
    /// Create a new graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get all the steps for the graph.
    pub fn steps(self) -> NodeSteps<B> {
        let mut map_drain = HashMap::new();
        self.execute_mut(|map| {
            std::mem::swap(&mut *map, &mut map_drain);
        });

        map_drain
    }

    /// Register a new step into the graph.
    pub fn register(self, id: &NodeID, ops: StepBoxed<B>) -> Self {
        self.execute_mut(|map| {
            map.insert(id.clone(), ops);
        })
    }

    /// Merge two graphs.
    pub fn merge(self, other: Self) -> Self {
        if Arc::ptr_eq(&self.steps, &other.steps) {
            return self.clone();
        }

        self.merge_different(other)
    }

    fn execute_mut<F: FnOnce(&mut NodeSteps<B>)>(mut self, func: F) -> Self {
        match Arc::get_mut(&mut self.steps) {
            Some(mutex) => {
                let map = mutex.get_mut().unwrap();
                func(map);
            }
            None => {
                // Only lock where there are multiple references to the graph.
                let mut map = self.steps.lock().unwrap();
                func(&mut map);
            }
        };

        self
    }

    fn merge_different(self, other: Self) -> Self {
        let mut map2 = other.steps();

        self.execute_mut(|map1| {
            if map1.len() > map2.len() {
                map1.extend(map2.into_iter());
            } else {
                let mut map_drain = HashMap::new();
                std::mem::swap(map1, &mut map_drain);
                map2.extend(map_drain.into_iter());
                std::mem::swap(map1, &mut map2);
            }
        })
    }
}
