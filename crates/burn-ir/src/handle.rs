use hashbrown::HashMap;

use crate::{BackendIr, TensorHandle, TensorId, TensorIr, TensorStatus};

/// Keep all [tensor handles](BackendIr::Handle) in one place and ensure that all resources
/// are used optimally.
#[derive(Default)]
pub struct HandleContainer<H> {
    handles: HashMap<TensorId, Handle<H>>,
    counter: u64,
}

impl<H: Clone> HandleContainer<H> {
    /// Fork the container, useful for autotune.
    pub fn fork(&self) -> Self {
        let mut handles = HashMap::with_capacity(self.handles.len());

        for (id, handle) in self.handles.iter() {
            handles.insert(*id, handle.clone());
        }

        Self {
            handles,
            counter: self.counter,
        }
    }
}

impl<H> core::fmt::Debug for HandleContainer<H> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HandleContainer")
            .field("handles", &self.handles.keys()) // only care about the IDs when debugging
            .field("counter", &self.counter)
            .finish()
    }
}

/// Backend [tensor handle](BackendIr::Handle) wrapper tracking their creation state
#[derive(Clone)]
pub enum Handle<H> {
    /// No [tensor handle](BackendIr::Handle) has been created yet
    NotInit,
    /// A [tensor handle](BackendIr::Handle) has been created
    Existing(H),
}

impl<H: Clone> HandleContainer<H> {
    /// Create a new HandleContainer
    pub fn new() -> Self {
        Self {
            handles: HashMap::new(),
            counter: 0,
        }
    }

    /// Register a handle for the given [tensor id](TensorId).
    pub fn register_handle(&mut self, id: TensorId, handle: H) {
        self.handles.insert(id, Handle::Existing(handle));
    }

    /// Whether an handle exists.
    pub fn has_handle(&mut self, id: &TensorId) -> bool {
        self.handles.contains_key(id)
    }

    /// Get the reference to a handle.
    pub fn get_handle_ref(&self, id: &TensorId) -> Option<&H> {
        self.handles
            .get(id)
            .filter(|h| !matches!(h, Handle::NotInit))
            .map(|h| match h {
                Handle::Existing(handle) => handle,
                Handle::NotInit => unreachable!(),
            })
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
            .unwrap_or_else(|| panic!("Should have handle for tensor {id:?}"));

        match handle {
            Handle::Existing(handle) => match status {
                TensorStatus::ReadOnly => {
                    self.handles.insert(id, Handle::Existing(handle.clone()));
                    handle
                }
                TensorStatus::ReadWrite => handle,
                TensorStatus::NotInit => panic!(
                    "Cannot get uninitialized tensor {id:?}. Tensor exist but with wrong status"
                ),
            },
            Handle::NotInit => panic!("Cannot get uninitialized handle {id:?}."),
        }
    }

    /// Get the tensor handle for the given [tensor intermediate representation](TensorIr).
    pub fn get_tensor_handle(&mut self, tensor: &TensorIr) -> TensorHandle<H> {
        TensorHandle {
            handle: self.get_handle(&tensor.id, &tensor.status),
            shape: tensor.shape.clone(),
        }
    }

    /// Get the [float tensor](burn_tensor::backend::Backend::FloatTensorPrimitive) corresponding to the
    /// given [tensor intermediate representation](TensorIr).
    pub fn get_float_tensor<B>(&mut self, tensor: &TensorIr) -> B::FloatTensorPrimitive
    where
        B: BackendIr<Handle = H>,
    {
        B::float_tensor(self.get_tensor_handle(tensor))
    }

    /// Get the [int tensor](burn_tensor::backend::Backend::IntTensorPrimitive) corresponding to the
    /// given [tensor intermediate representation](TensorIr).
    pub fn get_int_tensor<B>(&mut self, tensor: &TensorIr) -> B::IntTensorPrimitive
    where
        B: BackendIr<Handle = H>,
    {
        B::int_tensor(self.get_tensor_handle(tensor))
    }

    /// Get the [bool tensor](burn_tensor::backend::Backend::BoolTensorPrimitive) corresponding to the
    /// given [tensor intermediate representation](TensorIr).
    pub fn get_bool_tensor<B>(&mut self, tensor: &TensorIr) -> B::BoolTensorPrimitive
    where
        B: BackendIr<Handle = H>,
    {
        B::bool_tensor(self.get_tensor_handle(tensor))
    }

    /// Get the [quantized tensor](burn_tensor::backend::Backend::QuantizedTensorPrimitive) corresponding to the
    /// given [tensor intermediate representation](TensorIr).
    pub fn get_quantized_tensor<B>(&mut self, tensor: &TensorIr) -> B::QuantizedTensorPrimitive
    where
        B: BackendIr<Handle = H>,
    {
        B::quantized_tensor(self.get_tensor_handle(tensor))
    }

    /// Register a new [float tensor](burn_tensor::backend::Backend::FloatTensorPrimitive) with the corresponding [tensor id](TensorId).
    pub fn register_float_tensor<B>(&mut self, id: &TensorId, tensor: B::FloatTensorPrimitive)
    where
        B: BackendIr<Handle = H>,
    {
        let handle = B::float_tensor_handle(tensor);
        self.handles.insert(*id, Handle::Existing(handle));
    }

    /// Register a new [quantized tensor](burn_tensor::backend::Backend::QuantizedTensorPrimitive) with the corresponding [tensor ids](TensorId).
    pub fn register_quantized_tensor<B>(
        &mut self,
        id: &TensorId,
        tensor: B::QuantizedTensorPrimitive,
    ) where
        B: BackendIr<Handle = H>,
    {
        let handle = B::quantized_tensor_handle(tensor);
        self.handles.insert(*id, Handle::Existing(handle));
    }

    /// Register a new [int tensor](burn_tensor::backend::Backend::IntTensorPrimitive) with the corresponding [tensor id](TensorId).
    pub fn register_int_tensor<B>(&mut self, id: &TensorId, tensor: B::IntTensorPrimitive)
    where
        B: BackendIr<Handle = H>,
    {
        let handle = B::int_tensor_handle(tensor);
        self.handles.insert(*id, Handle::Existing(handle));
    }

    /// Register a new [bool tensor](burn_tensor::backend::Backend::BoolTensorPrimitive) with the corresponding [tensor id](TensorId).
    pub fn register_bool_tensor<B>(&mut self, id: &TensorId, tensor: B::BoolTensorPrimitive)
    where
        B: BackendIr<Handle = H>,
    {
        let handle = B::bool_tensor_handle(tensor);
        self.handles.insert(*id, Handle::Existing(handle));
    }

    /// Lazily create a new empty tensor and return its corresponding [tensor id](TensorId).
    pub fn create_tensor_uninit(&mut self) -> TensorId {
        let id = TensorId::new(self.counter);
        self.counter += 1;
        self.handles.insert(id, Handle::NotInit);
        id
    }

    /// Remove tensor handle from container.
    pub fn remove_handle(&mut self, id: TensorId) -> Option<Handle<H>> {
        self.handles.remove(&id)
    }

    /// Remove tensor handle from container if writable
    pub fn free(&mut self, tensor: &TensorIr) {
        match tensor.status {
            TensorStatus::ReadOnly => (),
            TensorStatus::NotInit => (),
            TensorStatus::ReadWrite => {
                self.handles.remove(&tensor.id);
            }
        };
    }

    /// Returns the number of handles.
    pub fn num_handles(&self) -> usize {
        self.handles.len()
    }
}
