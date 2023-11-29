use crate::client::FusionClient;
use burn_tensor::{
    backend::Backend,
    ops::{FloatElem, IntElem},
    Data, Reader, Shape,
};
use std::sync::Arc;

/// Tensor primitive for the [fusion backend](crate::FusionBackend) for all kind.
#[derive(Clone)]
pub struct FusionTensor<C: FusionClient> {
    /// Tensor id.
    pub id: Arc<TensorId>,
    /// The shape of the tensor.
    pub shape: Vec<usize>,
    /// The [fusion client](FusionClient).
    pub client: C,
    // Orphan means that a tensor is never converted into a description when it becomes `ReadWrite`.
    //
    // When a tensor is dropped and is still an orphan, we need to register it as such to avoid
    // memory leak. Otherwise, the cleanup is going to happen during a graph execution.
    pub(crate) is_orphan: bool,
}

impl<C: FusionClient> core::fmt::Debug for FusionTensor<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(
            format!(
                "{{ id: {:?}, shape: {:?}, should_drop: {:?}, backend: {:?}, device: {:?} }}",
                self.id,
                self.shape,
                self.is_orphan,
                <C::FusionBackend as Backend>::name(),
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
            is_orphan: true,
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

        if let TensorStatus::ReadWrite = status {
            self.is_orphan = false;
        }

        TensorDescription {
            status,
            shape: shape_out,
            id: self.id.as_ref().clone(),
        }
    }

    pub(crate) fn into_data<const D: usize>(self) -> Reader<Data<FloatElem<C::FusionBackend>, D>> {
        self.client
            .clone()
            .read_tensor_float(self.into_description())
    }

    pub(crate) fn int_into_data<const D: usize>(
        self,
    ) -> Reader<Data<IntElem<C::FusionBackend>, D>> {
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
        if !self.is_orphan {
            return;
        }

        match self.status() {
            TensorStatus::ReadWrite => {
                self.client.register_orphan(&self.id);
            }
            TensorStatus::ReadOnly => {}
            TensorStatus::NotInit => {}
        }
    }
}

/// The tensor unique identifier.
#[derive(Clone, Hash, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct TensorId {
    value: u64,
}

/// The status of the current tensor.
#[derive(Hash, Clone, Debug, PartialEq, Eq)]
pub enum TensorStatus {
    /// The tensor can be read, but not written.
    ReadOnly,
    /// The tensor can be mutated inplace.
    ReadWrite,
    /// No handle exists for that tensor.
    NotInit,
}

/// A tensor definition represents a snapshot of a tensor when it was used.
///
/// # Example
///
/// A tensor that is used multiple times has its status updated for each operation.
///
///   1. Status::NotInit
///   2. Status::ReadOnly
///   3. Status::ReadOnly
///   4. Status::ReadWrite
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct TensorDescription {
    /// The [tensor id](TensorId).
    pub id: TensorId,
    /// The shape of the tensor.
    pub shape: Vec<usize>,
    /// The [status](TensorStatus) of the tensor when it was used.
    pub status: TensorStatus,
}

impl TensorId {
    /// Create a new tensor id.
    pub fn new(value: u64) -> Self {
        Self { value }
    }
}
