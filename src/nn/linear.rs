use crate::module::Forward;
use crate::tensor::back::Backend;
use crate::tensor::{Distribution, Shape, Tensor};

pub struct LinearConfig {
    d_input: usize,
    d_output: usize,
}

pub struct Linear<B: Backend> {
    weight: Tensor<2, B>,
    bias: Option<Tensor<1, B>>,
}

impl<B: Backend> Linear<B> {
    pub fn new(config: &LinearConfig) -> Self {
        let weight = Tensor::random(
            Shape::new([config.d_input, config.d_output]),
            Distribution::Standard,
        );
        let bias = Some(Tensor::zeros(Shape::new([config.d_output])));

        Self { weight, bias }
    }
}

impl<B: Backend, const D: usize> Forward<&Tensor<D, B>, Tensor<D, B>> for Linear<B> {
    fn forward(&self, input: &Tensor<D, B>) -> Tensor<D, B> {
        let output = self.weight.unsqueeze().matmul(input);

        match &self.bias {
            Some(bias) => output + bias.unsqueeze(),
            None => output,
        }
    }
}
