use crate as burn;

use crate::config::Config;
use crate::module::Module;
use crate::module::{Forward, Param};
use crate::tensor::backend::Backend;
use crate::tensor::{Distribution, ElementConversion, Tensor};

#[derive(Config)]
pub struct EmbeddingConfig {
    n_embedding: usize,
    d_model: usize,
}

#[derive(Module, Debug)]
pub struct Embedding<B: Backend> {
    weight: Param<Tensor<B, 2>>,
}

impl<B: Backend> Embedding<B> {
    pub fn new(config: &EmbeddingConfig) -> Self {
        // Glorot init
        let start = -1.0 / f64::sqrt(config.d_model as f64);
        let end = 1.0 / f64::sqrt(config.d_model as f64);
        let distribution = Distribution::Uniform(start.to_elem(), end.to_elem());

        let weight = Tensor::random([config.n_embedding, config.d_model], distribution);

        Self {
            weight: Param::new(weight),
        }
    }
}

impl<B: Backend> Forward<Tensor<B::IntegerBackend, 2>, Tensor<B, 3>> for Embedding<B> {
    fn forward(&self, input: Tensor<B::IntegerBackend, 2>) -> Tensor<B, 3> {
        burn_tensor::module::embedding(&self.weight, &input)
    }
}
