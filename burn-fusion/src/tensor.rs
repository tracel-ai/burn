use crate::client::FusionClient;
use burn_tensor::{
    backend::Backend,
    ops::{FloatElem, IntElem},
    Data, Reader, Shape,
};
use std::sync::Arc;

#[derive(Clone)]
pub struct FusionTensor<C: FusionClient> {
    pub id: Arc<TensorId>,
    pub shape: Vec<usize>,
    pub client: C,
    pub should_drop: bool,
}

impl<C: FusionClient> core::fmt::Debug for FusionTensor<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(
            format!(
                "{{ id: {:?}, shape: {:?}, should_drop: {:?}, backend: {:?}, device: {:?} }}",
                self.id,
                self.shape,
                self.should_drop,
                <C::FusedBackend as Backend>::name(),
                self.client.device().clone().into(),
            )
            .as_str(),
        )
    }
}

impl<C: FusionClient> FusionTensor<C> {
    pub(crate) fn new(id: Arc<TensorId>, shape: Vec<usize>, client: C) -> Self {
        Self {
            id,
            shape,
            client,
            should_drop: true,
        }
    }
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
    pub(crate) fn into_description(mut self) -> TensorDescription {
        let status = self.status();
        let mut shape_out = Vec::new();
        core::mem::swap(&mut self.shape, &mut shape_out);

        match status {
            TensorStatus::ReadWrite => {
                // Used for the last time, so it's going to be dropped.
                self.should_drop = false;
            }
            _ => {}
        }

        TensorDescription {
            status,
            shape: shape_out,
            id: self.id.as_ref().clone(),
        }
    }

    pub(crate) fn into_data<const D: usize>(self) -> Reader<Data<FloatElem<C::FusedBackend>, D>> {
        self.client
            .clone()
            .read_tensor_float(self.into_description())
    }

    pub(crate) fn int_into_data<const D: usize>(self) -> Reader<Data<IntElem<C::FusedBackend>, D>> {
        self.client.clone().read_tensor_int(self.into_description())
    }

    pub(crate) fn bool_into_data<const D: usize>(self) -> Reader<Data<bool, D>> {
        self.client
            .clone()
            .read_tensor_bool(self.into_description())
    }
}

impl<C: FusionClient> Drop for FusionTensor<C> {
    fn drop(&mut self) {
        if !self.should_drop {
            return;
        }

        match self.status() {
            TensorStatus::ReadWrite => {
                self.client.drop_tensor(&self.id);
            }
            TensorStatus::ReadOnly => {}
            TensorStatus::NotInit => {}
        }
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
