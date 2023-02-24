use std::sync::{Arc, Mutex};

use burn_tensor::backend::Backend;
use dashmap::DashMap;

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
}
impl OpsMetadata {
    pub fn infer_requirement(&self, other: &Self) -> Requirement {
        self.requirement.infer(&other.requirement)
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct OpsID {
    value: String,
}

impl OpsID {
    pub fn new() -> Self {
        Self {
            value: nanoid::nanoid!(5).to_string(),
        }
    }
}

pub type OpsMetadataRef = Arc<OpsMetadata>;

#[derive(new, Debug)]
pub struct OpsMetadata {
    pub parents: Vec<OpsID>,
    pub order: usize,
    pub id: OpsID,
    pub requirement: Requirement,
}

pub type OpsBoxed<B> = Box<dyn Ops<B>>;
pub trait Ops<B: Backend>: Send + Sync + std::fmt::Debug {
    fn backward(self: Box<Self>, grads: &mut Gradients<B>);
    fn metadata(&self) -> OpsMetadataRef;
}

type SharedMap<B> = Arc<DashMap<OpsID, OpsBoxed<B>>>;

#[derive(Default, Clone)]
pub struct OpsMap<B: Backend> {
    map: Arc<Mutex<SharedMap<B>>>,
}

impl<B: Backend> std::fmt::Debug for OpsMap<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(format!("OpsMap<{:?}>", B::name()).as_str())
    }
}

impl<B: Backend> OpsMap<B> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn metadata(&self, id: &OpsID) -> Option<OpsMetadataRef> {
        let map = self.map.lock().unwrap();

        map.get(id).map(|ops| ops.metadata())
    }

    // TODO: Switch type to backward table before executing tons of operations in a row.
    pub fn pop(&self, id: &OpsID) -> Option<OpsBoxed<B>> {
        let map = self.map.lock().unwrap();

        map.remove(id).map(|ops| ops.1)
    }

    pub fn register(&self, id: &OpsID, ops: OpsBoxed<B>) {
        let map = self.map.lock().unwrap();

        map.insert(id.clone(), ops);
    }

    pub fn merge(&self, other: &Self) -> Self {
        if Arc::ptr_eq(&self.map, &other.map) {
            return self.clone();
        }

        self.merge_different(other);

        self.clone()
    }

    fn merge_different(&self, other: &Self) {
        let map1 = self.map.lock().unwrap();
        let mut map2 = other.map.lock().unwrap();
        let mut map_tmp = Arc::new(DashMap::new());

        std::mem::swap(&mut *map2, &mut map_tmp);
        Arc::try_unwrap(map_tmp)
            .unwrap()
            .into_iter()
            .for_each(|item| {
                map1.insert(item.0, item.1);
            });

        // Map1 and Map2 point to the same location.
        std::mem::swap(&mut *map2, &mut map1.clone());
    }
}
