use crate::client::FusionClient;
use burn_tensor::{ops::FloatElem, Data, Reader, Shape};
use std::sync::Arc;

#[derive(new, Clone, Debug)]
pub struct FusionTensor<C: FusionClient> {
    pub id: Arc<TensorId>,
    pub shape: Vec<usize>,
    pub client: C,
}

impl<C: FusionClient> FusionTensor<C> {
    pub(crate) fn shape<const D: usize>(&self) -> Shape<D> {
        Shape::from(self.shape.clone())
    }

    fn status(&self) -> TensorStatus {
        if Arc::strong_count(&self.id) <= 2 {
            TensorStatus::ReadWrite
        } else {
            TensorStatus::ReadOnly
        }
    }

    pub(crate) fn into_description(self) -> TensorDescription {
        TensorDescription {
            status: self.status(),
            shape: self.shape,
            id: self.id.as_ref().clone(),
        }
    }

    pub(crate) fn into_data<const D: usize>(self) -> Reader<Data<FloatElem<C::FusedBackend>, D>> {
        self.client.clone().read_float(self.into_description())
    }
}

#[derive(Clone, Hash, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct TensorId {
    value: u64,
}

#[derive(Clone, Debug)]
pub enum TensorStatus {
    ReadOnly,
    ReadWrite,
}

/// A tensor definition represent a snapshot of a tensor when it was used.
#[derive(Debug)]
pub struct TensorDescription {
    pub id: TensorId,
    pub shape: Vec<usize>,
    pub status: TensorStatus,
}

impl TensorId {
    pub(crate) fn new(value: u64) -> Self {
        Self { value }
    }
}
