use crate::{graph::FusedBackend, TensorDefinition, TensorId, TensorStatus};
use burn_tensor::Shape;
use std::{collections::HashMap, sync::Arc};

#[derive(Default)]
pub struct HandleContainer<B: FusedBackend> {
    handles: HashMap<TensorId, Handle<B>>,
    device: B::Device,
}

enum Handle<B: FusedBackend> {
    Empty,
    Existing(B::Handle),
}

pub enum TensorCreation {
    Empty { shape: Vec<usize> },
}

impl<B: FusedBackend> HandleContainer<B> {
    pub fn new(device_handle: B::HandleDevice) -> Self {
        Self {
            handles: HashMap::new(),
            device: device_handle.clone().into(),
        }
    }

    pub fn get_float_tensor<const D: usize>(
        &mut self,
        tensor: &TensorDefinition,
    ) -> B::TensorPrimitive<D> {
        match tensor.status {
            TensorStatus::ReadOnly => match self.handles.get(&tensor.id).unwrap() {
                Handle::Empty => {
                    let output = B::empty(Shape::from(tensor.shape.clone()), &self.device);
                    self.handles.insert(
                        tensor.id.clone(),
                        Handle::Existing(B::float_tensor_handle(output.clone())),
                    );
                    output
                }
                Handle::Existing(handle) => B::float_tensor(handle.clone()),
            },
            TensorStatus::ReadWrite => match self.handles.remove(&tensor.id).unwrap() {
                Handle::Empty => B::empty(Shape::from(tensor.shape.clone()), &self.device),
                Handle::Existing(handle) => B::float_tensor(handle),
            },
        }
    }

    pub fn register_float_tensor<const D: usize>(
        &mut self,
        id: &TensorId,
        tensor: B::TensorPrimitive<D>,
    ) {
        let handle = B::float_tensor_handle(tensor);
        self.handles.insert(id.clone(), Handle::Existing(handle));
    }

    pub fn create_empty(&mut self, shape: Vec<usize>) -> Arc<TensorId> {
        let id = TensorId::new();
        self.handles.insert(id.clone(), Handle::Empty);

        Arc::new(id)
    }

    pub fn cleanup(&mut self, tensor: &TensorDefinition) {
        match tensor.status {
            TensorStatus::ReadOnly => (),
            TensorStatus::ReadWrite => {
                self.handles.remove(&tensor.id);
            }
        }
    }
}
