use crate::tensor::back::ad::Backend;
use crate::tensor::{Gradients, Tensor};

pub trait Optimizer<B: Backend>: Send + Sync {
    fn update<const D: usize>(&mut self, tensor: &mut Tensor<B, D>, grads: &Gradients);
}
