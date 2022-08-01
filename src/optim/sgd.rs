use crate::optim::Optimizer;
use crate::tensor::back;
use crate::tensor::Gradients;
use crate::tensor::Tensor;

pub struct SGDOptimizer<B: back::ad::Backend> {
    learning_rate: <B::InnerBackend as back::Backend>::Elem,
}

impl<B: back::ad::Backend> Optimizer<B> for SGDOptimizer<B> {
    fn update<const D: usize>(&mut self, tensor: &mut Tensor<D, B>, grads: &Gradients) {
        let grad = tensor.grad(&grads).unwrap();
        let delta = grad.mul_scalar(&self.learning_rate);
        tensor.update(tensor.inner() + delta);
    }
}
