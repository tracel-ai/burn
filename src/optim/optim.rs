use crate::tensor::back::ad::Backend;
use crate::tensor::{Gradients, Tensor};

pub trait Optimizer: Send + Sync {
    type Backend: Backend;

    fn update<const D: usize>(&mut self, tensor: &mut Tensor<Self::Backend, D>, grads: &Gradients);
}
