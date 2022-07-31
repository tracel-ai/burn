use crate::module::Forward;
use crate::tensor::back::Backend;
use crate::tensor::Tensor;

pub struct Linear<B: Backend> {
    weight: Tensor<2, B>,
    bias: Option<Tensor<1, B>>,
}

impl<B: Backend, const D: usize> Forward<&Tensor<D, B>, Tensor<D, B>> for Linear<B> {
    fn forward(&self, input: &Tensor<D, B>) -> Tensor<D, B> {
        let output = self.weight.unsqueeze().matmul(input);

        let output = match &self.bias {
            Some(bias) => output + bias.unsqueeze(),
            None => output,
        };

        output
    }
}
