use crate::optim::Optimizer;
use crate::tensor::backend;
use crate::tensor::ElementConversion;
use crate::tensor::Gradients;
use crate::tensor::Tensor;

pub struct SGDOptimizer<B: backend::ADBackend> {
    learning_rate: <B::InnerBackend as backend::Backend>::Elem,
}

impl<B: backend::ADBackend> SGDOptimizer<B> {
    pub fn new(learning_rate: f64) -> Self {
        let learning_rate = learning_rate.to_elem();
        Self { learning_rate }
    }
}
impl<B: backend::ADBackend> Optimizer for SGDOptimizer<B> {
    type Backend = B;

    fn update<const D: usize>(&mut self, tensor: &mut Tensor<B, D>, grads: &Gradients) {
        let grad = tensor.grad(grads).unwrap();
        let delta = grad.mul_scalar(&self.learning_rate);
        tensor.update(tensor.inner() - delta);
    }
}
