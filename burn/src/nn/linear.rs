use crate as burn;

use crate::macros::config;
use crate::module::Module;
use crate::module::{Forward, Param};
use crate::tensor::backend::Backend;
use crate::tensor::{Distribution, ElementConversion, Shape, Tensor};
use std::ops::Deref;

config!(
    pub struct LinearConfig {
        pub d_input: usize,
        pub d_output: usize,
        pub bias: bool,
    }
);

#[derive(Module, Debug)]
pub struct Linear<B: Backend> {
    weight: Param<Tensor<B, 2>>,
    bias: Param<Option<Tensor<B, 1>>>,
}

impl<B: Backend> Linear<B> {
    pub fn new(config: &LinearConfig) -> Self {
        // Glorot init
        let start = -1.0 / f64::sqrt(config.d_input as f64);
        let end = 1.0 / f64::sqrt(config.d_input as f64);
        let distribution = Distribution::Uniform(start.to_elem(), end.to_elem());

        let weight = Tensor::random(Shape::new([config.d_input, config.d_output]), distribution);
        let bias = match config.bias {
            true => Some(Tensor::zeros(Shape::new([config.d_output]))),
            false => None,
        };

        Self {
            weight: Param::new(weight),
            bias: Param::new(bias),
        }
    }
}

impl<B: Backend, const D: usize> Forward<Tensor<B, D>, Tensor<B, D>> for Linear<B> {
    fn forward(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let output = input.matmul(&self.weight.unsqueeze());

        match self.bias.deref() {
            Some(bias) => output + bias.unsqueeze(),
            None => output,
        }
    }
}
