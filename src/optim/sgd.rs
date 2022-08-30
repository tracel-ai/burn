use crate::optim::Optimizer;
use crate::tensor::back;
use crate::tensor::ElementConversion;
use crate::tensor::Gradients;
use crate::tensor::Tensor;

pub struct SGDOptimizer<B: back::ad::Backend> {
    learning_rate: <B::InnerBackend as back::Backend>::Elem,
}

impl<B: back::ad::Backend> SGDOptimizer<B> {
    pub fn new(learning_rate: f64) -> Self {
        let learning_rate = learning_rate.to_elem();
        Self { learning_rate }
    }
}
impl<B: back::ad::Backend> Optimizer for SGDOptimizer<B> {
    type Backend = B;

    fn update<const D: usize>(&mut self, tensor: &mut Tensor<B, D>, grads: &Gradients) {
        let grad = tensor.grad(&grads).unwrap();
        let delta = grad.mul_scalar(&self.learning_rate);
        tensor.update(tensor.inner() - delta);
    }
}
