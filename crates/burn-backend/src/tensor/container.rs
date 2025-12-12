use alloc::boxed::Box;
use core::any::Any;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(not(feature = "std"))]
use hashbrown::HashMap;

#[cfg(feature = "std")]
use std::collections::HashMap;

use crate::{TensorPrimitive, backend::Backend};

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
    pub fn get<B>(&self, id: &ID) -> Option<TensorPrimitive<B>>
    where
        B: Backend,
    {
        let grad = self.tensors.get(id)?;

        let tensor = grad
            .downcast_ref::<TensorPrimitive<B>>()
            // .map(|primitive| Tensor::<B, D>::from_primitive(primitive.clone()))
            .unwrap();

        Some(tensor.clone())
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
    pub fn remove<B>(&mut self, id: &ID) -> Option<TensorPrimitive<B>>
    where
        B: Backend,
    {
        self.tensors
            .remove(id)
            .map(|item| *item.downcast::<TensorPrimitive<B>>().unwrap())
        // .map(|primitive| Tensor::from_primitive(*primitive))
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
