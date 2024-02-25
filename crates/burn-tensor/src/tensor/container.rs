#[cfg(not(feature = "std"))]
use hashbrown::HashMap;

use core::{fmt::Debug, hash::Hash};
#[cfg(feature = "std")]
use std::collections::HashMap;

use crate::{backend::Backend, DynRankTensor, Tensor};

/// Contains tensors of arbitrary dimension, as [`DynRankTensor`]s.
#[derive(Debug)]
pub struct TensorContainer<Id, B: Backend> {
    tensors: HashMap<Id, DynRankTensor<B>>,
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
    pub fn get<const D: usize>(&self, id: &Id) -> Option<Tensor<B, D>> {
        let dyn_rank_tensor = self.tensors.get(id)?.clone();

        Some(dyn_rank_tensor.into())
    }

    /// Register a new tensor for the given ID.
    ///
    /// # Notes
    ///
    /// If a tensor is already registered for the given ID, it will be replaced.
    pub fn register<const D: usize>(&mut self, id: Id, value: Tensor<B, D>) {
        self.tensors.insert(id, value.into());
    }

    /// Remove a tensor for the given ID and returns it.
    pub fn remove<const D: usize>(&mut self, id: &Id) -> Option<Tensor<B, D>> {
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
    pub fn into_inner(self) -> HashMap<Id, DynRankTensor<B>> {
        self.tensors
    }

    /// Creates a new [`TensorContainer`] from a [`HashMap`] of [`DynRankTensor`]s, which is the
    /// internal representation of it.
    pub fn from_inner(tensors: HashMap<Id, DynRankTensor<B>>) -> Self {
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
