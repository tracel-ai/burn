use alloc::boxed::Box;
use core::any::Any;

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;

#[cfg(feature = "std")]
use std::collections::HashMap;

use crate::{TensorPrimitive, backend::Backend};

/// Error type for tensor container operations.
#[derive(Debug)]
pub enum TensorContainerError {
    /// The tensor with the given ID was not found.
    NotFound,

    /// Downcast mismatch when retrieving tensor.
    /// If you are trying to retrieve the gradients for a given parameter id, make sure to use the inner backend.
    /// Gradients are not stored on the autodiff backend.
    DowncastError,
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
    pub fn get<B>(&self, id: &ID) -> Result<TensorPrimitive<B>, TensorContainerError>
    where
        B: Backend,
    {
        match self.tensors.get(id) {
            Some(grad) => match grad.downcast_ref::<TensorPrimitive<B>>() {
                Some(tensor) => Ok(tensor.clone()),
                None => Err(TensorContainerError::DowncastError),
            },
            None => Err(TensorContainerError::NotFound),
        }
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

    /// Remove a tensor for the given ID and returns it.
    pub fn remove<B>(&mut self, id: &ID) -> Result<TensorPrimitive<B>, TensorContainerError>
    where
        B: Backend,
    {
        match self.tensors.remove(id) {
            Some(tensor) => match tensor.downcast::<TensorPrimitive<B>>() {
                Ok(tensor) => Ok(*tensor),
                Err(_uncast) => Err(TensorContainerError::DowncastError),
            },
            None => Err(TensorContainerError::NotFound),
        }
    }

    /// The number of tensors registered.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// If any tensor is contained.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
