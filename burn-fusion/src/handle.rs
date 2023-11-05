use crate::{graph::FusedBackend, TensorId};
use std::{collections::HashMap, sync::Arc};

#[derive(Default)]
pub struct HandleContainer<B: FusedBackend> {
    handles: HashMap<TensorId, B::Handle>,
    references: HashMap<TensorId, Arc<TensorId>>,
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
            references: HashMap::new(),
            device,
        }
    }

    pub fn get(&mut self, id: &TensorId) -> HandleResult<B::Handle> {
        if let Some(tensor) = self.references.get(id) {
            let count = Arc::strong_count(&tensor);

            if count == 0 {
                HandleResult::NotInitialized
            } else if count <= 2 {
                let handle = self.handles.remove(&id).unwrap();
                self.references.remove(id);
                HandleResult::ReadWrite(handle)
            } else {
                let handle = self.handles.get(&id).unwrap().clone();
                HandleResult::ReadOnly(handle)
            }
        } else {
            panic!("No handle");
        }
    }

    pub fn get_float_tensor<const D: usize>(&mut self, id: &TensorId) -> B::TensorPrimitive<D> {
        match self.get(id) {
            HandleResult::ReadOnly(handle) => B::float_tensor(handle),
            HandleResult::ReadWrite(handle) => B::float_tensor(handle),
            HandleResult::NotInitialized => panic!(),
        }
    }

    pub fn register_float_tensor<const D: usize>(
        &mut self,
        id: &TensorId,
        tensor: B::TensorPrimitive<D>,
    ) {
        let handle = B::float_tensor_handle(tensor);
        self.handles.insert(id.clone(), handle);
    }

    pub fn create(&mut self, shape: Vec<usize>, handle: B::Handle) -> Arc<TensorId> {
        let id = TensorId::new();
        let reference = Arc::new(id.clone());

        self.handles.insert(id.clone(), handle);
        self.references.insert(id, reference.clone());

        reference
    }

    pub fn not_initialized(&mut self, shape: Vec<usize>) -> Arc<TensorId> {
        let id = TensorId::new();
        let reference = Arc::new(id.clone());

        self.references.insert(id, reference.clone());

        reference
    }
}
