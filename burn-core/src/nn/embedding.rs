use alloc::{format, vec::Vec};
use burn_tensor::Int;

use crate as burn;

use crate::config::Config;
use crate::module::Module;
use crate::module::Param;
use crate::tensor::backend::Backend;
use crate::tensor::{Distribution, Tensor};

/// Configuration to create an [Embedding](Embedding) layer.
#[derive(Config)]
pub struct EmbeddingConfig {
    /// The number of embedding vectors.
    n_embedding: usize,
    /// The size of each vector.
    d_model: usize,
}

/// Lookup table to store a fix number of vectors.
///
/// # Params
///
/// - weight: Matrix of shape `[n_embedding, d_model]` initialized from a normal distribution:
///     `N(0, 1)`
#[derive(Module, Debug)]
pub struct Embedding<B: Backend> {
    weight: Param<Tensor<B, 2>>,
}

impl<B: Backend> Embedding<B> {
    /// Create the module from the given configuration.
    pub fn new(config: &EmbeddingConfig) -> Self {
        let weight = Tensor::random(
            [config.n_embedding, config.d_model],
            Distribution::Normal(0.0, 1.0),
        )
        .require_grad();

        Self {
            weight: Param::from(weight),
        }
    }

    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - input: [batch_size, seq_length]
    /// - output: [batch_size, d_model]
    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        burn_tensor::module::embedding(self.weight.val(), input)
    }
}
