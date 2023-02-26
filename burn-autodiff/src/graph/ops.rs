use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use burn_common::id::IdGenerator;
use burn_tensor::backend::Backend;

use crate::grads::Gradients;

#[derive(Debug, Clone, Copy)]
pub enum Requirement {
    Grad,
    GradInBackward,
    None,
}

impl Requirement {
    pub fn infer(&self, other: &Self) -> Self {
        match self {
            Self::Grad => return Self::GradInBackward,
            Self::GradInBackward => return Self::GradInBackward,
            Self::None => (),
        }

        match other {
            Self::Grad => Self::GradInBackward,
            Self::GradInBackward => Self::GradInBackward,
            Self::None => Self::None,
        }
    }

    pub fn from_metadata(metadata: &[MetadataRef]) -> Self {
        metadata
            .iter()
            .map(|metadata| metadata.requirement)
            .reduce(|acc, requirement| requirement.infer(&acc))
            .unwrap_or(Requirement::None)
    }
}

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

    pub fn extract(&self) -> HashMap<OpsID, Node<B>> {
        let mut map = self.ops.lock().unwrap();
        let mut map_drain = HashMap::new();
        std::mem::swap(&mut *map, &mut map_drain);

        map_drain
    }

    pub fn register(&self, id: &OpsID, ops: Node<B>) {
        let mut map = self.ops.lock().unwrap();

        map.insert(id.clone(), ops);
    }

    pub fn merge(&self, other: &Self) -> Self {
        if Arc::ptr_eq(&self.ops, &other.ops) {
            return self.clone();
        }

        self.merge_different(other);

        self.clone()
    }

    fn merge_different(&self, other: &Self) {
        let mut map1 = self.ops.lock().unwrap();
        let mut map2 = other.ops.lock().unwrap();
        let mut map_drain = HashMap::new();

        std::mem::swap(&mut *map2, &mut map_drain);
        map_drain.into_iter().for_each(|item| {
            map1.insert(item.0, item.1);
        });
    }
}
