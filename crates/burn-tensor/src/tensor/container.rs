use hashbrown::HashMap;

use core::{fmt::Debug, hash::Hash};
use hashbrown::hash_map::{DefaultHashBuilder, Entry};

use crate::{DynPrimBackend, DynTensor, Tensor};

/// Contains tensors of arbitrary dimension, as [`DynTensor`]s.
#[derive(Debug, Clone)]
pub struct TensorContainer<Id, P> {
    tensors: HashMap<Id, DynTensor<P>>,
}

impl<Id, P> TensorContainer<Id, P>
where
    Id: Hash + PartialEq + Eq + Debug,
{
    /// Create an empty container.
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
        }
    }

    /// Register a new tensor for the given ID.
    ///
    /// # Notes
    ///
    /// If a tensor is already registered for the given ID, it will be replaced.
    pub fn register<B: DynPrimBackend<P>, const D: usize>(
        &mut self,
        id: Id,
        value: Tensor<B, D>,
    ) {
        self.tensors
            .insert(id, DynTensor::from(value));
    }

    /// Returns the underlying [HashMap] entry for a given ID.
    pub fn entry(&mut self, id: Id) -> Entry<'_, Id, DynTensor<P>, DefaultHashBuilder> {
        self.tensors.entry(id)
    }

    /// Remove a tensor for the given ID and returns it.
    pub fn remove<B: DynPrimBackend<P>, const D: usize>(
        &mut self,
        id: &Id,
    ) -> Option<Tensor<B, D>> {
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
    pub fn into_inner(self) -> HashMap<Id, DynTensor<P>> {
        self.tensors
    }

    /// Creates a new [`TensorContainer`] from a [`HashMap`] of [`DynRankTensor`]s, which is the
    /// internal representation of it.
    pub fn from_inner(
        tensors: HashMap<Id, DynTensor<P>>,
    ) -> Self {
        Self { tensors }
    }
}

impl<Id, P> TensorContainer<Id, P>
    where
        P: Clone,
        Id: Hash + PartialEq + Eq + Debug,
{
    /// Get a tensor with the given ID.
    pub fn get<B: DynPrimBackend<P>, const D: usize>(
        &self,
        id: &Id,
    ) -> Option<Tensor<B, D>> {
        let dyn_tensor = self.tensors.get(id)?.clone();

        Some(dyn_tensor.into())
    }
}