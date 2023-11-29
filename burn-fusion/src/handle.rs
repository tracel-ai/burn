use crate::{FusionBackend, TensorDescription, TensorId, TensorStatus};
use burn_tensor::Shape;
use std::{collections::HashMap, sync::Arc};

/// Keep all [tensor handles](FusionBackend::Handle) in one place and ensure that all resources
/// are used optimally.
#[derive(Default)]
pub struct HandleContainer<B: FusionBackend> {
    handles: HashMap<TensorId, Handle<B>>,
    counter: u64,
    pub(crate) handles_orphan: Vec<TensorId>,
    /// The device on which all tensors are held.
    pub device: B::Device,
}

enum Handle<B: FusionBackend> {
    NotInit,
    Existing(B::Handle),
}

impl<B: FusionBackend> HandleContainer<B> {
    pub(crate) fn new(device_handle: B::FusionDevice) -> Self {
        Self {
            handles: HashMap::new(),
            handles_orphan: Vec::new(),
            counter: 0,
            device: device_handle.clone().into(),
        }
    }

    /// Register a handle for the given [tensor id](TensorId).
    pub fn register_handle(&mut self, id: TensorId, handle: B::Handle) {
        self.handles.insert(id, Handle::Existing(handle));
    }

    /// Get the handle for the given [tensor id](TensorId).
    pub fn get_handle(&mut self, tensor: &TensorDescription) -> B::Handle {
        let (id, handle) = self
            .handles
            .remove_entry(&tensor.id)
            .unwrap_or_else(|| panic!("Should have handle for tensor {:?}", tensor.id));

        match handle {
            Handle::Existing(handle) => match tensor.status {
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

    /// Get the [float tensor](burn_tensor::backend::Backend::TensorPrimitive) corresponding to the
    /// given [tensor description](TensorDescription).
    pub fn get_float_tensor<const D: usize>(
        &mut self,
        tensor: &TensorDescription,
    ) -> B::TensorPrimitive<D> {
        B::float_tensor(self.get_handle(tensor), Shape::from(&tensor.shape))
    }

    /// Get the [int tensor](burn_tensor::backend::Backend::IntTensorPrimitive) corresponding to the
    /// given [tensor description](TensorDescription).
    pub fn get_int_tensor<const D: usize>(
        &mut self,
        tensor: &TensorDescription,
    ) -> B::IntTensorPrimitive<D> {
        B::int_tensor(self.get_handle(tensor), Shape::from(&tensor.shape))
    }

    /// Get the [bool tensor](burn_tensor::backend::Backend::BoolTensorPrimitive) corresponding to the
    /// given [tensor description](TensorDescription).
    pub fn get_bool_tensor<const D: usize>(
        &mut self,
        tensor: &TensorDescription,
    ) -> B::BoolTensorPrimitive<D> {
        B::bool_tensor(self.get_handle(tensor), Shape::from(&tensor.shape))
    }

    /// Register a new [float tensor](burn_tensor::backend::Backend::TensorPrimitive) with the corresponding [tensor id](TensorId).
    pub fn register_float_tensor<const D: usize>(
        &mut self,
        id: &TensorId,
        tensor: B::TensorPrimitive<D>,
    ) {
        let handle = B::float_tensor_handle(tensor);
        self.handles.insert(id.clone(), Handle::Existing(handle));
    }

    /// Register a new [int tensor](burn_tensor::backend::Backend::IntTensorPrimitive) with the corresponding [tensor id](TensorId).
    pub fn register_int_tensor<const D: usize>(
        &mut self,
        id: &TensorId,
        tensor: B::IntTensorPrimitive<D>,
    ) {
        let handle = B::int_tensor_handle(tensor);
        self.handles.insert(id.clone(), Handle::Existing(handle));
    }

    /// Register a new [bool tensor](burn_tensor::backend::Backend::BoolTensorPrimitive) with the corresponding [tensor id](TensorId).
    pub fn register_bool_tensor<const D: usize>(
        &mut self,
        id: &TensorId,
        tensor: B::BoolTensorPrimitive<D>,
    ) {
        let handle = B::bool_tensor_handle(tensor);
        self.handles.insert(id.clone(), Handle::Existing(handle));
    }

    /// Lazily create a new empty tensor and return its corresponding [tensor id](TensorId).
    pub fn create_tensor_uninit(&mut self) -> Arc<TensorId> {
        let id = TensorId::new(self.counter);
        self.counter += 1;
        self.handles.insert(id.clone(), Handle::NotInit);

        Arc::new(id)
    }

    pub(crate) fn cleanup(&mut self, tensor: &TensorDescription) {
        match tensor.status {
            TensorStatus::ReadOnly => (),
            TensorStatus::NotInit => (),
            TensorStatus::ReadWrite => {
                self.handles.remove(&tensor.id);
            }
        }
    }

    pub(crate) fn cleanup_orphans(&mut self) {
        for id in self.handles_orphan.drain(..) {
            self.handles.remove(&id);
        }
    }
}
