use crate::{graph::FusedBackend, Client};
use burn_tensor::Shape;
use std::sync::{atomic::AtomicU64, Arc};

#[derive(new, Clone, Debug)]
pub struct FusionTensor<B: FusedBackend> {
    pub shape: Vec<usize>,
    pub id: Arc<TensorId>,
    pub client: Client<B>,
    pub device: B::HandleDevice,
}

impl<B: FusedBackend> FusionTensor<B> {
    pub(crate) fn shape<const D: usize>(&self) -> Shape<D> {
        Shape::from(self.shape.clone())
    }
    pub fn can_mut(&self) -> bool {
        Arc::strong_count(&self.id) <= 2
    }

    pub(crate) fn into_definition(self) -> TensorDefinition {
        TensorDefinition {
            id: self.id.as_ref().clone(),
            shape: self.shape,
        }
    }

    pub(crate) fn to_definition(&self) -> TensorDefinition {
        TensorDefinition {
            id: self.id.as_ref().clone(),
            shape: self.shape.clone(),
        }
    }
}

const ID_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Hash, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct TensorId {
    value: u64,
}

#[derive(Clone, Debug)]
pub struct TensorDefinition {
    pub id: TensorId,
    pub shape: Vec<usize>,
}

impl TensorId {
    pub(crate) fn new() -> Self {
        let id = ID_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        Self { value: id.into() }
    }
}
