use crate::client::FusionClient;
use burn_tensor::{
    ops::{FloatElem, IntElem},
    Data, Reader, Shape,
};
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
        if Arc::strong_count(&self.id) <= 1 {
            TensorStatus::ReadWrite
        } else {
            TensorStatus::ReadOnly
        }
    }

    /// Description to be used when using an uninitialized tensor as output.
    pub(crate) fn to_description_out(&self) -> TensorDescription {
        TensorDescription {
            status: TensorStatus::NotInit,
            shape: self.shape.clone(),
            id: self.id.as_ref().clone(),
        }
    }

    /// Description to be used when using an initialized tensor used as input.
    pub(crate) fn into_description(self) -> TensorDescription {
        let status = self.status();
        TensorDescription {
            status: self.status(),
            shape: self.shape,
            id: self.id.as_ref().clone(),
        }
    }

    pub(crate) fn into_data<const D: usize>(self) -> Reader<Data<FloatElem<C::FusedBackend>, D>> {
        self.client.clone().read_float(self.into_description())
    }

    pub(crate) fn int_into_data<const D: usize>(self) -> Reader<Data<IntElem<C::FusedBackend>, D>> {
        self.client.clone().read_int(self.into_description())
    }

    pub(crate) fn bool_into_data<const D: usize>(self) -> Reader<Data<bool, D>> {
        self.client.clone().read_bool(self.into_description())
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
    NotInit,
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
