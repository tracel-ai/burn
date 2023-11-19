use alloc::boxed::Box;
use core::any::Any;

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;

#[cfg(feature = "std")]
use std::collections::HashMap;

use crate::{backend::Backend, Tensor};

/// Contains tensor of arbitrary dimension.
#[derive(Debug)]
pub struct TensorContainer<ID> {
    tensors: HashMap<ID, Box<dyn Any + Send + Sync>>,
}

impl<ID> Default for TensorContainer<ID>
where
    ID: core::hash::Hash + PartialEq + Eq + core::fmt::Debug,
{
    fn default() -> Self {
        Self::new()
    }
}

type TensorPrimitive<B, const D: usize> = <B as Backend>::TensorPrimitive<D>;

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
    pub fn get<B, const D: usize>(&self, id: &ID) -> Option<Tensor<B, D>>
    where
        B: Backend,
    {
        let grad = match self.tensors.get(id) {
            Some(grad) => grad,
            None => return None,
        };

        let tensor = grad
            .downcast_ref::<TensorPrimitive<B, D>>()
            .map(|primitive| Tensor::<B, D>::from_primitive(primitive.clone()))
            .unwrap();

        Some(tensor)
    }

    /// Register a new tensor for the given ID.
    ///
    /// # Notes
    ///
    /// If a tensor is already registered for the given ID, it will be replaced.
    pub fn register<B, const D: usize>(&mut self, id: ID, value: Tensor<B, D>)
    where
        B: Backend,
    {
        self.tensors.insert(id, Box::new(value.into_primitive()));
    }

    /// Remove a tensor for the given ID and returns it.
    pub fn remove<B, const D: usize>(&mut self, id: &ID) -> Option<Tensor<B, D>>
    where
        B: Backend,
    {
        self.tensors
            .remove(id)
            .map(|item| item.downcast::<TensorPrimitive<B, D>>().unwrap())
            .map(|primitive| Tensor::from_primitive(*primitive))
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
