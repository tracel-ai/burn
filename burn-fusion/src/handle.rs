use crate::{graph::FusedBackend, Client, FusionTensor, TensorId};
use std::{collections::HashMap, sync::Arc};

#[derive(Default)]
pub struct HandleContainer<B: FusedBackend> {
    handles: HashMap<TensorId, B::Handle>,
    tensors: HashMap<TensorId, FusionTensor<B>>,
    device: B::HandleDevice,
}

pub enum HandleResult<Handle> {
    ReadOnly(Handle),
    ReadWrite(Handle),
    NotInitialized,
}

pub enum TensorCreation {
    Empty { shape: Vec<usize> },
}

impl<B: FusedBackend> HandleContainer<B> {
    pub fn new(device: B::HandleDevice) -> Self {
        Self {
            handles: HashMap::new(),
            tensors: HashMap::new(),
            device,
        }
    }
    pub fn get(&mut self, id: &TensorId) -> HandleResult<B::Handle> {
        if let Some(tensor) = self.tensors.get(id) {
            let handle = self.handles.get(&id).unwrap().clone();

            if tensor.can_mut() {
                HandleResult::ReadWrite(handle)
            } else {
                HandleResult::ReadOnly(handle)
            }
        } else {
            panic!("No handle");
        }
    }

    pub fn create(
        &mut self,
        shape: Vec<usize>,
        handle: B::Handle,
        client: Client<B>,
    ) -> FusionTensor<B> {
        let id = TensorId::new();
        let reference = Arc::new(id.clone());
        let tensor = FusionTensor::new(shape, reference, client, self.device.clone());

        self.handles.insert(id.clone(), handle);
        self.tensors.insert(id, tensor.clone());

        tensor
    }

    pub fn not_initialized(&mut self, shape: Vec<usize>) -> (B::HandleDevice, Arc<TensorId>) {
        let id = TensorId::new();
        let reference = Arc::new(id.clone());

        // self.tensors.insert(id, tensor.clone());

        (self.device.clone(), reference)
    }
}
