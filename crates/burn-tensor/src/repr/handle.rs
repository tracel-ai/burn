use crate::{
    repr::{
        backend::ReprBackend,
        tensor::{TensorDescription, TensorId, TensorStatus},
    },
    Shape,
};
use std::{collections::HashMap, sync::Arc};

/// Keep all [tensor handles](ReprBackend::Handle) in one place and ensure that all resources
/// are used optimally.
#[derive(Default)]
pub struct HandleContainer<H> {
    handles: HashMap<TensorId, Handle<H>>,
    counter: u64,
    /// Handle candidates to be freed.
    pub handles_orphan: Vec<TensorId>,
}

/// Backend [tensor handle](ReprBackend::Handle) wrapper tracking their creation state
pub enum Handle<H> {
    /// No [tensor handle](ReprBackend::Handle) has been created yet
    NotInit,
    /// A [tensor handle](ReprBackend::Handle) has been created
    Existing(H),
}

impl<H: Clone> HandleContainer<H> {
    /// Create a new HandleContainer
    pub fn new() -> Self {
        Self {
            handles: HashMap::new(),
            handles_orphan: Vec::new(),
            counter: 0,
        }
    }

    /// Register a handle for the given [tensor id](TensorId).
    pub fn register_handle(&mut self, id: TensorId, handle: H) {
        self.handles.insert(id, Handle::Existing(handle));
    }

    /// Get the handle for the given [tensor id](TensorId). The status is used to determine if the
    /// tensor should be popped out of the current tensor map, necessary for inplace operations.
    ///
    /// # Warnings
    ///
    /// Make sure the status corresponds to the operation you want to execute the handle on,
    /// otherwise you might remove a tensor handle that will be required in the future.
    pub fn get_handle(&mut self, id: &TensorId, status: &TensorStatus) -> H {
        let (id, handle) = self
            .handles
            .remove_entry(id)
            .unwrap_or_else(|| panic!("Should have handle for tensor {:?}", id));

        match handle {
            Handle::Existing(handle) => match status {
                TensorStatus::ReadOnly => {
                    self.handles.insert(id, Handle::Existing(handle.clone()));
                    handle
                }
                TensorStatus::ReadWrite => handle,
                TensorStatus::NotInit => panic!("Cannot get uninitialized tensor."),
            },
            Handle::NotInit => panic!("Cannot get uninitialized handle."),
        }
    }

    /// Get the [float tensor](crate::backend::Backend::FloatTensorPrimitive) corresponding to the
    /// given [tensor description](TensorDescription).
    pub fn get_float_tensor<B, const D: usize>(
        &mut self,
        tensor: &TensorDescription,
    ) -> B::FloatTensorPrimitive<D>
    where
        B: ReprBackend<Handle = H>,
    {
        B::float_tensor::<D>(
            self.get_handle(&tensor.id, &tensor.status),
            Shape::from(&tensor.shape),
        )
    }

    /// Get the [int tensor](crate::backend::Backend::IntTensorPrimitive) corresponding to the
    /// given [tensor description](TensorDescription).
    pub fn get_int_tensor<B, const D: usize>(
        &mut self,
        tensor: &TensorDescription,
    ) -> B::IntTensorPrimitive<D>
    where
        B: ReprBackend<Handle = H>,
    {
        B::int_tensor::<D>(
            self.get_handle(&tensor.id, &tensor.status),
            Shape::from(&tensor.shape),
        )
    }

    /// Get the [bool tensor](crate::backend::Backend::BoolTensorPrimitive) corresponding to the
    /// given [tensor description](TensorDescription).
    pub fn get_bool_tensor<B, const D: usize>(
        &mut self,
        tensor: &TensorDescription,
    ) -> B::BoolTensorPrimitive<D>
    where
        B: ReprBackend<Handle = H>,
    {
        B::bool_tensor::<D>(
            self.get_handle(&tensor.id, &tensor.status),
            Shape::from(&tensor.shape),
        )
    }

    /// Register a new [float tensor](crate::backend::Backend::FloatTensorPrimitive) with the corresponding [tensor id](TensorId).
    pub fn register_float_tensor<B, const D: usize>(
        &mut self,
        id: &TensorId,
        tensor: B::FloatTensorPrimitive<D>,
    ) where
        B: ReprBackend<Handle = H>,
    {
        let handle = B::float_tensor_handle::<D>(tensor);
        self.handles.insert(*id, Handle::Existing(handle));
    }

    /// Register a new [int tensor](crate::backend::Backend::IntTensorPrimitive) with the corresponding [tensor id](TensorId).
    pub fn register_int_tensor<B, const D: usize>(
        &mut self,
        id: &TensorId,
        tensor: B::IntTensorPrimitive<D>,
    ) where
        B: ReprBackend<Handle = H>,
    {
        let handle = B::int_tensor_handle::<D>(tensor);
        self.handles.insert(*id, Handle::Existing(handle));
    }

    /// Register a new [bool tensor](crate::backend::Backend::BoolTensorPrimitive) with the corresponding [tensor id](TensorId).
    pub fn register_bool_tensor<B, const D: usize>(
        &mut self,
        id: &TensorId,
        tensor: B::BoolTensorPrimitive<D>,
    ) where
        B: ReprBackend<Handle = H>,
    {
        let handle = B::bool_tensor_handle::<D>(tensor);
        self.handles.insert(*id, Handle::Existing(handle));
    }

    /// Lazily create a new empty tensor and return its corresponding [tensor id](TensorId).
    pub fn create_tensor_uninit(&mut self) -> Arc<TensorId> {
        let id = TensorId::new(self.counter);
        self.counter += 1;
        self.handles.insert(id, Handle::NotInit);

        Arc::new(id)
    }

    /// Remove tensor handle from container if writable
    pub fn free(&mut self, tensor: &TensorDescription) {
        match tensor.status {
            TensorStatus::ReadOnly => (),
            TensorStatus::NotInit => (),
            TensorStatus::ReadWrite => {
                self.handles.remove(&tensor.id);
            }
        }
    }

    /// Remove tensor handle from container if not in use
    pub fn free_orphans(&mut self, remaining: &[&TensorId]) {
        let mut handles_orphan = Vec::new();

        // TODO: Optimization => Change the for loop order depending of the length of each.
        for id in self.handles_orphan.drain(..) {
            if remaining.contains(&&id) {
                handles_orphan.push(id);
            } else {
                self.handles.remove(&id);
            }
        }

        self.handles_orphan = handles_orphan;
    }
}
