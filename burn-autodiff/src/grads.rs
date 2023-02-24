use burn_tensor::{backend::Backend, container::TensorContainer, Tensor};

use crate::tensor::ADTensor;

pub type GradID = String;
#[derive(Default)]
pub struct Gradients<B: Backend> {
    container: TensorContainer<B, GradID>,
}

impl<B: Backend> Gradients<B> {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn consume<const D: usize>(&mut self, tensor: &ADTensor<B, D>) -> B::TensorPrimitive<D> {
        self.container
            .get(&tensor.metadata.id.value)
            .map(|tensor| tensor.into_primitive())
            .unwrap_or_else(|| B::zeros(B::shape(&tensor.primitive), &B::device(&tensor.primitive)))
    }
    pub fn get<const D: usize>(&self, tensor: &ADTensor<B, D>) -> Option<B::TensorPrimitive<D>> {
        self.container
            .get(&tensor.metadata.id.value)
            .map(|tensor| tensor.into_primitive())
    }

    pub fn update<const D: usize>(&mut self, tensor: ADTensor<B, D>, value: B::TensorPrimitive<D>) {
        println!("Register tensor {:?}", tensor.metadata.id);
        // TODO: Check if exists first and add it
        self.container.register(
            tensor.metadata.id.value.clone(),
            Tensor::from_primitive(value),
        )
    }
}
