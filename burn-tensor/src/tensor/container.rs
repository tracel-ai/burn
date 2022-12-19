use crate::{backend::Backend, Tensor};
use std::{any::Any, collections::HashMap};

#[derive(Default, Debug)]
pub struct TensorContainer<B: Backend, ID> {
    tensors: HashMap<ID, Box<dyn Any + Send + Sync>>,
    _b: B,
}

impl<B, ID> TensorContainer<B, ID>
where
    B: Backend,
    ID: std::hash::Hash + PartialEq + Eq,
{
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
            _b: B::default(),
        }
    }

    pub fn get<const D: usize>(&self, id: &ID) -> Option<Tensor<B, D>> {
        let grad = match self.tensors.get(id) {
            Some(grad) => grad,
            None => return None,
        };

        let tensor = grad
            .downcast_ref()
            .map(|primitive: &B::TensorPrimitive<D>| Tensor::from_primitive(primitive.clone()));
        tensor
    }

    pub fn register<const D: usize>(&mut self, id: ID, value: Tensor<B, D>) {
        self.tensors.insert(id, Box::new(value.into_primitive()));
    }

    pub fn remove<const D: usize>(&mut self, id: &ID) -> Option<Tensor<B, D>> {
        self.tensors
            .remove(id)
            .map(|item| item.downcast::<B::TensorPrimitive<D>>().unwrap())
            .map(|primitive| Tensor::from_primitive(*primitive))
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
