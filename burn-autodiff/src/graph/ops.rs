use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use burn_common::id::IdGenerator;
use burn_tensor::backend::Backend;

use crate::grads::Gradients;
use crate::graph::Requirement;

impl Metadata {
    pub fn clone_if_require_grad(self: &Arc<Self>) -> Option<MetadataRef> {
        match self.requirement {
            Requirement::None => None,
            _ => Some(self.clone()),
        }
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct OpsID {
    pub(crate) value: String,
}

impl OpsID {
    pub fn new() -> Self {
        Self {
            value: IdGenerator::generate(),
        }
    }
}

impl Default for OpsID {
    fn default() -> Self {
        Self::new()
    }
}

pub type MetadataRef = Arc<Metadata>;

#[derive(new, Debug)]
pub struct Metadata {
    pub parents: Vec<OpsID>,
    pub order: usize,
    pub id: OpsID,
    pub requirement: Requirement,
}

pub trait Backward<B: Backend>: Send + Sync + std::fmt::Debug {
    fn backward(self: Box<Self>, grads: &mut Gradients<B>);
    fn metadata(&self) -> MetadataRef;
}

pub type Node<B> = Box<dyn Backward<B>>;
#[derive(Default, Clone)]
pub struct Graph<B: Backend> {
    ops: Arc<Mutex<HashMap<OpsID, Node<B>>>>,
}

impl<B: Backend> std::fmt::Debug for Graph<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(format!("OpsMap<{:?}>", B::name()).as_str())
    }
}

impl<B: Backend> Graph<B> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn extract(self) -> HashMap<OpsID, Node<B>> {
        let mut map_drain = HashMap::new();
        self.execute_mut(|map| {
            std::mem::swap(&mut *map, &mut map_drain);
        });

        map_drain
    }

    pub fn register(self, id: &OpsID, ops: Node<B>) -> Self {
        self.execute_mut(|map| {
            map.insert(id.clone(), ops);
        })
    }

    pub fn merge(self, other: Self) -> Self {
        if Arc::ptr_eq(&self.ops, &other.ops) {
            return self.clone();
        }

        self.merge_different(other)
    }

    fn execute_mut<F: FnOnce(&mut HashMap<OpsID, Node<B>>)>(mut self, func: F) -> Self {
        match Arc::get_mut(&mut self.ops) {
            Some(mutex) => {
                let map = mutex.get_mut().unwrap();
                func(map);
            }
            None => {
                // Only lock where there are multiple references to the graph.
                let mut map = self.ops.lock().unwrap();
                func(&mut map);
            }
        };

        self
    }

    fn merge_different(self, other: Self) -> Self {
        let mut map2 = other.extract();

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
