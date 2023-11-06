use crate::{FloatElem, FusedBackend, TensorDescription, TensorId, TensorStatus};
use burn_tensor::{Data, Shape};
use std::{collections::HashMap, sync::Arc};

#[derive(Default)]
pub struct HandleContainer<B: FusedBackend> {
    handles: HashMap<TensorId, Handle<B>>,
    pub device: B::Device,
}

enum Handle<B: FusedBackend> {
    Empty,
    DataFloat(Vec<FloatElem<B>>),
    Existing(B::Handle),
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
        tensor: &TensorDescription,
    ) -> B::TensorPrimitive<D> {
        let (id, handle) = self.handles.remove_entry(&tensor.id).unwrap();

        if let Handle::Existing(handle) = handle {
            match tensor.status {
                TensorStatus::ReadOnly => {
                    self.handles.insert(id, Handle::Existing(handle.clone()));
                    return B::float_tensor(handle);
                }
                TensorStatus::ReadWrite => {
                    return B::float_tensor(handle);
                }
            }
        }

        let output = match handle {
            Handle::Empty => B::empty(Shape::from(tensor.shape.clone()), &self.device),
            Handle::DataFloat(values) => B::from_data(
                Data::new(values, Shape::from(tensor.shape.clone())),
                &self.device,
            ),
            Handle::Existing(handle) => B::float_tensor(handle),
        };

        if let TensorStatus::ReadOnly = tensor.status {
            self.handles
                .insert(id, Handle::Existing(B::float_tensor_handle(output.clone())));
        }

        output
    }

    pub fn register_float_tensor<const D: usize>(
        &mut self,
        id: &TensorId,
        tensor: B::TensorPrimitive<D>,
    ) {
        let handle = B::float_tensor_handle(tensor);
        self.handles.insert(id.clone(), Handle::Existing(handle));
    }

    pub fn create_emtpy(&mut self) -> Arc<TensorId> {
        let id = TensorId::new();
        self.handles.insert(id.clone(), Handle::Empty);

        Arc::new(id)
    }

    pub fn create_float(&mut self, values: Vec<FloatElem<B>>) -> Arc<TensorId> {
        let id = TensorId::new();
        self.handles.insert(id.clone(), Handle::DataFloat(values));

        Arc::new(id)
    }

    pub fn cleanup(&mut self, tensor: &TensorDescription) {
        match tensor.status {
            TensorStatus::ReadOnly => (),
            TensorStatus::ReadWrite => {
                self.handles.remove(&tensor.id);
            }
        }
    }
}
