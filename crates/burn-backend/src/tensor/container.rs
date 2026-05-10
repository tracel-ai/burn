use alloc::boxed::Box;
use alloc::format;
use alloc::string::String;
use core::any::Any;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(not(feature = "std"))]
use hashbrown::HashMap;

#[cfg(feature = "std")]
use std::collections::HashMap;

use thiserror::Error;

use crate::{TensorPrimitive, backend::Backend};

/// Errors that can be returned by [`TensorContainer`] accessors.
#[derive(Debug, Error)]
pub enum TensorContainerError {
    /// No tensor is registered for the given identifier.
    #[error("tensor container: no entry for id {id}")]
    NotFound {
        /// Debug-formatted identifier of the missing entry.
        id: String,
    },
    /// A tensor is registered for the given identifier, but the stored [`TensorPrimitive`]
    /// is for a different backend than the one requested.
    ///
    /// The most common cause is passing `B: AutodiffBackend` to a gradient accessor that
    /// expects `B::InnerBackend` (or vice-versa). Gradients are stored on the inner backend
    /// because the autodiff wrapper doesn't track itself.
    #[error(
        "tensor container: type mismatch on tensor with id {id}; the stored TensorPrimitive does not match the requested Backend. \
         Most commonly: pass `B::InnerBackend` instead of `B: AutodiffBackend` when accessing gradients."
    )]
    TypeMismatch {
        /// Debug-formatted identifier where the mismatch was detected.
        id: String,
    },
}

/// Contains tensor of arbitrary dimension.
#[derive(Debug)]
pub struct TensorContainer<ID> {
    tensors: HashMap<ID, Box<dyn Any + Send>>,
}

impl<ID> Default for TensorContainer<ID>
where
    ID: core::hash::Hash + PartialEq + Eq + core::fmt::Debug,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<ID> TensorContainer<ID>
where
    ID: core::hash::Hash + PartialEq + Eq + core::fmt::Debug,
{
    /// Create an empty container.
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
        }
    }

    /// Get a tensor with the given ID.
    ///
    /// Returns [`TensorContainerError::NotFound`] if no tensor is registered for the ID and
    /// [`TensorContainerError::TypeMismatch`] if a tensor is registered for a different backend.
    pub fn get<B>(&self, id: &ID) -> Result<TensorPrimitive<B>, TensorContainerError>
    where
        B: Backend,
    {
        let grad = self
            .tensors
            .get(id)
            .ok_or_else(|| TensorContainerError::NotFound {
                id: format!("{id:?}"),
            })?;

        let tensor = grad.downcast_ref::<TensorPrimitive<B>>().ok_or_else(|| {
            TensorContainerError::TypeMismatch {
                id: format!("{id:?}"),
            }
        })?;

        Ok(tensor.clone())
    }

    /// Get a mutable reference to the tensor with the given ID.
    ///
    /// Returns [`TensorContainerError::NotFound`] if no tensor is registered for the ID and
    /// [`TensorContainerError::TypeMismatch`] if a tensor is registered for a different backend.
    pub fn get_mut_ref<B>(
        &mut self,
        id: &ID,
    ) -> Result<&mut TensorPrimitive<B>, TensorContainerError>
    where
        B: Backend,
    {
        let id_str = format!("{id:?}");
        let grad = self
            .tensors
            .get_mut(id)
            .ok_or_else(|| TensorContainerError::NotFound { id: id_str.clone() })?;

        grad.downcast_mut::<TensorPrimitive<B>>()
            .ok_or(TensorContainerError::TypeMismatch { id: id_str })
    }

    /// Register a new tensor for the given ID.
    ///
    /// # Notes
    ///
    /// If a tensor is already registered for the given ID, it will be replaced.
    pub fn register<B>(&mut self, id: ID, value: TensorPrimitive<B>)
    where
        B: Backend,
    {
        self.tensors.insert(id, Box::new(value));
    }

    /// Remove a tensor for the given ID and return it.
    ///
    /// Returns [`TensorContainerError::NotFound`] if no tensor is registered for the ID and
    /// [`TensorContainerError::TypeMismatch`] if a tensor is registered for a different backend.
    /// On `TypeMismatch`, the entry is left in the container so a subsequent correctly-typed
    /// access can still retrieve it.
    pub fn remove<B>(&mut self, id: &ID) -> Result<TensorPrimitive<B>, TensorContainerError>
    where
        B: Backend,
    {
        // Peek before removing — a downcast on the boxed entry would consume it on the
        // failure path, leaking the tensor. Probe the type first; only commit to removal
        // once we know it will succeed.
        let entry = self
            .tensors
            .get(id)
            .ok_or_else(|| TensorContainerError::NotFound {
                id: format!("{id:?}"),
            })?;

        if !entry.is::<TensorPrimitive<B>>() {
            return Err(TensorContainerError::TypeMismatch {
                id: format!("{id:?}"),
            });
        }

        // Safe to remove: type check above guarantees the downcast succeeds.
        let boxed = self.tensors.remove(id).expect("entry was just observed");
        Ok(*boxed
            .downcast::<TensorPrimitive<B>>()
            .expect("type was just observed to match"))
    }

    /// The number of tensors registered.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// If any tensor is contained.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get id of every tensor in the container
    pub fn ids(&self) -> Vec<&ID> {
        self.tensors.keys().collect()
    }
}
