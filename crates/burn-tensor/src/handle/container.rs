use crate::{
    backend::Backend,
    handle::{TensorDescription, TensorId, TensorStatus},
    ops::{BoolTensor, FloatTensor, IntTensor},
    Shape,
};
use std::{collections::HashMap, sync::Arc};

/// Backend extension trait that allows an existing [backend](Backend) to store its tensors in a .
pub trait HandleContainerBackend: Backend {
    /// The type that can be used to point to a tensor of any kind.
    type Handle: Sync + Send + Clone;

    /// Convert a [handle](HandleContainerBackend::Handle) to a [float tensor](Backend::FloatTensorPrimitive).
    fn float_tensor<const D: usize>(handle: Self::Handle, shape: Shape<D>) -> FloatTensor<Self, D>;
    /// Convert a [handle](HandleContainerBackend::Handle) to an [int tensor](Backend::IntTensorPrimitive).
    fn int_tensor<const D: usize>(handle: Self::Handle, shape: Shape<D>) -> IntTensor<Self, D>;
    /// Convert a [handle](HandleContainerBackend::Handle) to a [bool tensor](Backend::BoolTensorPrimitive).
    fn bool_tensor<const D: usize>(handle: Self::Handle, shape: Shape<D>) -> BoolTensor<Self, D>;

    /// Convert a [float tensor](Backend::FloatTensorPrimitive) to a [handle](HandleContainerBackend::Handle).
    fn float_tensor_handle<const D: usize>(tensor: FloatTensor<Self, D>) -> Self::Handle;
    /// Convert an [int tensor](Backend::IntTensorPrimitive) to a [handle](HandleContainerBackend::Handle).
    fn int_tensor_handle<const D: usize>(tensor: IntTensor<Self, D>) -> Self::Handle;
    /// Convert a [bool tensor](Backend::BoolTensorPrimitive) to a [handle](HandleContainerBackend::Handle).
    fn bool_tensor_handle<const D: usize>(tensor: BoolTensor<Self, D>) -> Self::Handle;
}

/// Keep all [tensor handles](HandleContainerBackend::Handle) in one place and ensure that all resources
/// are used optimally.
#[derive(Default)]
pub struct HandleContainer<B: HandleContainerBackend> {
    handles: HashMap<TensorId, Handle<B>>,
    counter: u64,
    /// --ADDED-- Handle candidates to be freed.
    pub handles_orphan: Vec<TensorId>,
    /// The device on which all tensors are held.
    pub device: B::Device,
}

/// --ADDED-- Backend [tensor handle](HandleContainerBackend::Handle) wrapper tracking their creation state
pub enum Handle<B: Backend + HandleContainerBackend> {
    /// --ADDED-- No [tensor handle](HandleContainerBackend::Handle) has been created yet
    NotInit,
    /// --ADDED-- A [tensor handle](HandleContainerBackend::Handle) has been created
    Existing(B::Handle),
}

impl<B: HandleContainerBackend> HandleContainer<B> {
    /// Create a new HandleContainer
    pub fn new(device_handle: B::Device) -> Self {
        Self {
            handles: HashMap::new(),
            handles_orphan: Vec::new(),
            counter: 0,
            device: device_handle.clone(),
        }
    }

    /// Register a handle for the given [tensor id](TensorId).
    pub fn register_handle(&mut self, id: TensorId, handle: B::Handle) {
        self.handles.insert(id, Handle::Existing(handle));
    }

    /// Get the handle for the given [tensor id](TensorId). The status is used to determine if the
    /// tensor should be popped out of the current tensor map, necessary for inplace operations.
    ///
    /// # Warnings
    ///
    /// Make sure the status corresponds to the operation you want to execute the handle on,
    /// otherwise you might remove a tensor handle that will be required in the future.
    pub fn get_handle(&mut self, id: &TensorId, status: &TensorStatus) -> B::Handle {
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

    /// Get the [float tensor](Backend::FloatTensorPrimitive) corresponding to the
    /// given [tensor description](TensorDescription).
    pub fn get_float_tensor<const D: usize>(
        &mut self,
        tensor: &TensorDescription,
    ) -> B::FloatTensorPrimitive<D> {
        B::float_tensor::<D>(
            self.get_handle(&tensor.id, &tensor.status),
            Shape::from(&tensor.shape),
        )
    }

    /// Get the [int tensor](Backend::IntTensorPrimitive) corresponding to the
    /// given [tensor description](TensorDescription).
    pub fn get_int_tensor<const D: usize>(
        &mut self,
        tensor: &TensorDescription,
    ) -> B::IntTensorPrimitive<D> {
        B::int_tensor::<D>(
            self.get_handle(&tensor.id, &tensor.status),
            Shape::from(&tensor.shape),
        )
    }

    /// Get the [bool tensor](Backend::BoolTensorPrimitive) corresponding to the
    /// given [tensor description](TensorDescription).
    pub fn get_bool_tensor<const D: usize>(
        &mut self,
        tensor: &TensorDescription,
    ) -> B::BoolTensorPrimitive<D> {
        B::bool_tensor::<D>(
            self.get_handle(&tensor.id, &tensor.status),
            Shape::from(&tensor.shape),
        )
    }

    /// Register a new [float tensor](Backend::FloatTensorPrimitive) with the corresponding [tensor id](TensorId).
    pub fn register_float_tensor<const D: usize>(
        &mut self,
        id: &TensorId,
        tensor: B::FloatTensorPrimitive<D>,
    ) {
        let handle = B::float_tensor_handle::<D>(tensor);
        self.handles.insert(*id, Handle::Existing(handle));
    }

    /// Register a new [int tensor](Backend::IntTensorPrimitive) with the corresponding [tensor id](TensorId).
    pub fn register_int_tensor<const D: usize>(
        &mut self,
        id: &TensorId,
        tensor: B::IntTensorPrimitive<D>,
    ) {
        let handle = B::int_tensor_handle::<D>(tensor);
        self.handles.insert(*id, Handle::Existing(handle));
    }

    /// Register a new [bool tensor](Backend::BoolTensorPrimitive) with the corresponding [tensor id](TensorId).
    pub fn register_bool_tensor<const D: usize>(
        &mut self,
        id: &TensorId,
        tensor: B::BoolTensorPrimitive<D>,
    ) {
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

    /// --ADDED-- Remove tensor handle from container if writable
    pub fn free(&mut self, tensor: &TensorDescription) {
        match tensor.status {
            TensorStatus::ReadOnly => (),
            TensorStatus::NotInit => (),
            TensorStatus::ReadWrite => {
                self.handles.remove(&tensor.id);
            }
        }
    }

    /// --ADDED-- Remove tensor handle from container if not in use
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
