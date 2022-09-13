use crate::tensor::backend::ADBackend;
use crate::tensor::{Gradients, Tensor};

pub trait Optimizer: Send + Sync {
    type Backend: ADBackend;

    fn update<const D: usize>(&mut self, tensor: &mut Tensor<Self::Backend, D>, grads: &Gradients);
}
