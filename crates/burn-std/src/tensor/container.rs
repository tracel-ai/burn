use alloc::boxed::Box;
use core::any::{Any, type_name};

/// Error returned when accessing a tensor registered in a [`TensorContainer`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorContainerError {
    /// No tensor is registered for the given ID.
    NotFound,
    /// A tensor is registered for the given ID, but with a different type than requested.
    TypeMismatch {
        /// The fully qualified name of the requested type.
        requested: &'static str,
    },
}

impl core::fmt::Display for TensorContainerError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotFound => write!(f, "No tensor is registered for the given ID."),
            Self::TypeMismatch { requested } => write!(
                f,
                "Tensor type mismatch: requested `{requested}`. Make sure the requested tensor type matches the registered tensor type."
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for TensorContainerError {}

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
    pub fn get<T: Clone + Send + 'static>(&self, id: &ID) -> Result<T, TensorContainerError> {
        let grad = self.tensors.get(id).ok_or(TensorContainerError::NotFound)?;

        grad.downcast_ref::<T>()
            .cloned()
            .ok_or(TensorContainerError::TypeMismatch {
                requested: type_name::<T>(),
            })
    }

    /// Get a mutable reference to the tensor with the given ID.
    pub fn get_mut_ref<T: Clone + Send + 'static>(
        &mut self,
        id: &ID,
    ) -> Result<&mut T, TensorContainerError> {
        let grad = self
            .tensors
            .get_mut(id)
            .ok_or(TensorContainerError::NotFound)?;

        grad.downcast_mut::<T>()
            .ok_or(TensorContainerError::TypeMismatch {
                requested: type_name::<T>(),
            })
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
    pub fn remove<T: Clone + Send + 'static>(
        &mut self,
        id: &ID,
    ) -> Result<T, TensorContainerError> {
        let grad = self.tensors.get(id).ok_or(TensorContainerError::NotFound)?;

        if grad.downcast_ref::<T>().is_none() {
            return Err(TensorContainerError::TypeMismatch {
                requested: type_name::<T>(),
            });
        }

        self.tensors
            .remove(id)
            .ok_or(TensorContainerError::NotFound)?
            .downcast::<T>()
            .map(|tensor| *tensor)
            .map_err(|_| TensorContainerError::TypeMismatch {
                requested: type_name::<T>(),
            })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_returns_registered_tensor() {
        let mut container = TensorContainer::<u64>::new();
        container.register(1, 42i32);

        assert_eq!(container.get::<i32>(&1), Ok(42));
    }

    #[test]
    fn get_missing_id_returns_not_found() {
        let container = TensorContainer::<u64>::new();

        assert_eq!(
            container.get::<i32>(&1),
            Err(TensorContainerError::NotFound)
        );
    }

    #[test]
    fn get_type_mismatch_returns_error_without_panicking() {
        let mut container = TensorContainer::<u64>::new();
        container.register(1, 42i32);

        let result = container.get::<i64>(&1);

        assert_eq!(
            result,
            Err(TensorContainerError::TypeMismatch {
                requested: type_name::<i64>()
            })
        );
        assert_eq!(container.get::<i32>(&1), Ok(42));
    }

    #[test]
    fn get_mut_ref_type_mismatch_returns_error_without_panicking() {
        let mut container = TensorContainer::<u64>::new();
        container.register(1, 42i32);

        let result = container.get_mut_ref::<i64>(&1);

        assert_eq!(
            result.err(),
            Some(TensorContainerError::TypeMismatch {
                requested: type_name::<i64>()
            })
        );
        *container.get_mut_ref::<i32>(&1).unwrap() += 1;
        assert_eq!(container.get::<i32>(&1), Ok(43));
    }

    #[test]
    fn remove_returns_tensor_once_then_not_found() {
        let mut container = TensorContainer::<u64>::new();
        container.register(1, 42i32);

        assert_eq!(container.remove::<i32>(&1), Ok(42));
        assert_eq!(
            container.remove::<i32>(&1),
            Err(TensorContainerError::NotFound)
        );
    }

    #[test]
    fn remove_type_mismatch_keeps_tensor_registered() {
        let mut container = TensorContainer::<u64>::new();
        container.register(1, 42i32);

        let result = container.remove::<i64>(&1);

        assert_eq!(
            result,
            Err(TensorContainerError::TypeMismatch {
                requested: type_name::<i64>()
            })
        );
        assert_eq!(container.get::<i32>(&1), Ok(42));
    }
}
