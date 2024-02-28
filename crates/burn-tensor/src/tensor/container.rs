#[cfg(not(feature = "std"))]
use hashbrown::HashMap;

use core::{fmt::Debug, hash::Hash};
#[cfg(feature = "std")]
use std::collections::HashMap;

use crate::{backend::Backend, DynTensor, Tensor};

/// Contains tensors of arbitrary dimension, as [`DynTensor`]s.
#[derive(Debug)]
pub struct TensorContainer<Id, B: Backend> {
    tensors: HashMap<Id, DynTensor<B>>,
}

impl<Id, B: Backend> TensorContainer<Id, B>
where
    Id: Hash + PartialEq + Eq + Debug,
{
    /// Create an empty container.
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
        }
    }

    /// Get a tensor with the given ID.
    pub fn get<BOut: Backend<DynTensorPrimitive = B::DynTensorPrimitive>, const D: usize>(
        &self,
        id: &Id,
    ) -> Option<Tensor<BOut, D>> {
        let dyn_tensor = self.tensors.get(id)?.clone();

        Some(dyn_tensor.as_backend::<BOut>().into())
    }

    /// Register a new tensor for the given ID.
    ///
    /// # Notes
    ///
    /// If a tensor is already registered for the given ID, it will be replaced.
    pub fn register<BIn: Backend<DynTensorPrimitive = B::DynTensorPrimitive>, const D: usize>(
        &mut self,
        id: Id,
        value: Tensor<BIn, D>,
    ) {
        self.tensors
            .insert(id, DynTensor::<BIn>::from(value).as_backend::<B>());
    }

    /// Remove a tensor for the given ID and returns it.
    pub fn remove<BOut: Backend<DynTensorPrimitive = B::DynTensorPrimitive>, const D: usize>(
        &mut self,
        id: &Id,
    ) -> Option<Tensor<BOut, D>> {
        self.tensors.remove(id).map(Into::into)
    }

    /// The number of tensors registered.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// If any tensor is contained.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Convert into the internal representation of the [`TensorContainer`].
    pub fn into_inner(self) -> HashMap<Id, DynTensor<B>> {
        self.tensors
    }

    /// Creates a new [`TensorContainer`] from a [`HashMap`] of [`DynRankTensor`]s, which is the
    /// internal representation of it.
    pub fn from_inner(tensors: HashMap<Id, DynTensor<B>>) -> Self {
        Self { tensors }
    }
}

impl<Id, B: Backend> Default for TensorContainer<Id, B>
where
    Id: Hash + PartialEq + Eq + Debug,
{
    fn default() -> Self {
        Self::new()
    }
}
