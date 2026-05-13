use alloc::boxed::Box;
use core::any::Any;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(not(feature = "std"))]
use hashbrown::HashMap;

#[cfg(feature = "std")]
use std::collections::HashMap;

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
    pub fn get<T: Clone + Send + 'static>(&self, id: &ID) -> Option<T> {
        let grad = self.tensors.get(id)?;

        let tensor = grad.downcast_ref::<T>().unwrap();

        Some(tensor.clone())
    }

    /// Get a mutable reference to the tensor with the given ID.
    pub fn get_mut_ref<T: Clone + Send + 'static>(&mut self, id: &ID) -> Option<&mut T> {
        let grad = self.tensors.get_mut(id)?;

        let tensor = grad.downcast_mut::<T>().unwrap();

        Some(tensor)
    }

    /// Register a new tensor for the given ID.
    ///
    /// # Notes
    ///
    /// If a tensor is already registered for the given ID, it will be replaced.
    pub fn register<T: Clone + Send + 'static>(&mut self, id: ID, value: T) {
        self.tensors.insert(id, Box::new(value));
    }

    /// Remove a tensor for the given ID and returns it.
    pub fn remove<T: Clone + Send + 'static>(&mut self, id: &ID) -> Option<T> {
        self.tensors
            .remove(id)
            .map(|item| *item.downcast::<T>().unwrap())
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
