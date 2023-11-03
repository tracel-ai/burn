use crate::{graph::FusedBackend, Client, FusionTensor, TensorId};
use std::{collections::HashMap, sync::Arc};

#[derive(Default)]
pub struct HandleContainer<B: FusedBackend> {
    handles: HashMap<TensorId, B::Handle>,
    tensors: HashMap<TensorId, FusionTensor<B>>,
}

pub enum HandleResult<Handle> {
    ReadOnly(Handle),
    ReadWrite(Handle),
}

impl<B: FusedBackend> HandleContainer<B> {
    pub fn new() -> Self {
        Self {
            handles: HashMap::new(),
            tensors: HashMap::new(),
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
        let tensor = FusionTensor::new(shape, reference, client);

        self.handles.insert(id.clone(), handle);
        self.tensors.insert(id, tensor.clone());

        tensor
    }
}
