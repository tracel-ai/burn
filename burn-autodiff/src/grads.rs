use burn_tensor::{backend::Backend, container::TensorContainer};

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
        todo!()
    }
    pub fn get<const D: usize>(&self, tensor: &ADTensor<B, D>) -> Option<B::TensorPrimitive<D>> {
        todo!()
    }

    pub fn update<const D: usize>(&mut self, tensor: ADTensor<B, D>, value: B::TensorPrimitive<D>) {
        todo!()
    }
}
