use alloc::boxed::Box;
use core::any::Any;
use core::marker::PhantomData;

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;

#[cfg(feature = "std")]
use std::collections::HashMap;

use crate::{backend::Backend, Tensor};

/// Contains tensor of arbitrary dimension.
#[derive(Debug)]
pub struct TensorContainer<B: Backend, ID> {
    tensors: HashMap<ID, Box<dyn Any + Send + Sync>>,
    phantom: PhantomData<B>,
}

impl<B, ID> Default for TensorContainer<B, ID>
where
    B: Backend,
    ID: core::hash::Hash + PartialEq + Eq,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<B, ID> TensorContainer<B, ID>
where
    B: Backend,
    ID: core::hash::Hash + PartialEq + Eq,
{
    /// Create an empty container.
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
            phantom: PhantomData::default(),
        }
    }

    /// Get a tensor with the given ID.
    pub fn get<const D: usize>(&self, id: &ID) -> Option<Tensor<B, D>> {
        let grad = match self.tensors.get(id) {
            Some(grad) => grad,
            None => return None,
        };

        let tensor = grad
            .downcast_ref()
            .map(|primitive: &B::TensorPrimitive<D>| Tensor::from_primitive(primitive.clone()));
        tensor
    }

    /// Register a new tensor for the given ID.
    ///
    /// # Notes
    ///
    /// If a tensor is already registered for the given ID, it will be replaced.
    pub fn register<const D: usize>(&mut self, id: ID, value: Tensor<B, D>) {
        self.tensors.insert(id, Box::new(value.into_primitive()));
    }

    /// Remove a tensor for the given ID and returns it.
    pub fn remove<const D: usize>(&mut self, id: &ID) -> Option<Tensor<B, D>> {
        self.tensors
            .remove(id)
            .map(|item| item.downcast::<B::TensorPrimitive<D>>().unwrap())
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
