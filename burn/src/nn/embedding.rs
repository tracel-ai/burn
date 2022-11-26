use crate as burn;

use crate::config::Config;
use crate::module::Module;
use crate::module::Param;
use crate::tensor::backend::Backend;
use crate::tensor::{Distribution, ElementConversion, Tensor};

/// Configuration to create an [Embedding](Embedding) layer.
#[derive(Config)]
pub struct EmbeddingConfig {
    /// The number of embedding vectors.
    n_embedding: usize,
    /// The size of each vector.
    d_model: usize,
}

/// Lookup table to store a fix number of vectors.
#[derive(Module, Debug)]
pub struct Embedding<B: Backend> {
    weight: Param<Tensor<B, 2>>,
}

impl<B: Backend> Embedding<B> {
    /// Create the module from the given configuration.
    pub fn new(config: &EmbeddingConfig) -> Self {
        let start = -1.0 / f64::sqrt(config.d_model as f64);
        let end = 1.0 / f64::sqrt(config.d_model as f64);
        let distribution = Distribution::Uniform(start.to_elem(), end.to_elem());
        let weight = Tensor::random([config.n_embedding, config.d_model], distribution);

        Self {
            weight: Param::new(weight),
        }
    }

    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - input: [batch_size, seq_length]
    /// - output: [batch_size, d_model]
    pub fn forward(&self, input: Tensor<B::IntegerBackend, 2>) -> Tensor<B, 3> {
        burn_tensor::module::embedding(&self.weight, &input)
    }
}
