use burn_tensor::backend::Backend;
use spin::Mutex;
use std::{collections::HashMap, sync::Arc};

use crate::grads::Gradients;

use super::{NodeID, NodeRef};

/// Backward step for reverse mode autodiff.
pub trait Step: Send + Sync + std::fmt::Debug {
    type Backend: Backend;

    /// Executes the step and consumes it.
    fn step(self: Box<Self>, grads: &mut Gradients<Self::Backend>);

    /// The node associated to the step.
    fn node(&self) -> NodeRef;
}

pub type StepBoxed<B> = Box<dyn Step<Backend = B>>;
pub type NodeSteps<B> = HashMap<NodeID, StepBoxed<B>>;

/// Graph data structure.
///
/// The graph contains the [node steps](Step), which can be access by [node id](NodeID).
#[derive(Default, Clone, Debug)]
pub struct Graph<B: Backend> {
    steps: Arc<Mutex<NodeSteps<B>>>,
}

impl<B: Backend> Graph<B> {
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
            return self;
        }

        self.merge_different(other)
    }

    fn execute_mut<F: FnOnce(&mut NodeSteps<B>)>(mut self, func: F) -> Self {
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
        let mut map2 = other.steps();

        self.execute_mut(|map1| {
            if map1.len() > map2.len() {
                map1.extend(map2);
            } else {
                let mut map_drain = HashMap::new();
                std::mem::swap(map1, &mut map_drain);
                map2.extend(map_drain);
                std::mem::swap(map1, &mut map2);
            }
        })
    }
}
