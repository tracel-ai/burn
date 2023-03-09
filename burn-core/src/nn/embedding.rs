use alloc::{format, vec::Vec};
use burn_tensor::Int;

use crate as burn;

use super::Initializer;
use crate::config::Config;
use crate::module::Module;
use crate::module::Param;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;

/// Configuration to create an [Embedding](Embedding) layer.
#[derive(Config)]
pub struct EmbeddingConfig {
    /// The number of embedding vectors.
    n_embedding: usize,
    /// The size of each vector.
    d_model: usize,
    /// The type of function used to initialize neural network parameters
    #[config(default = "Initializer::Normal(0.0,1.0)")]
    pub initializer: Initializer,
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
        let weight = config
            .initializer
            .init([config.n_embedding, config.d_model])
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

#[cfg(test)]
mod tests {
    use burn_tensor::Data;

    use super::*;
    pub type TB = burn_ndarray::NdArrayBackend<f32>;

    #[test]
    fn initializer_default() {
        TB::seed(0);
        let config = EmbeddingConfig::new(100, 10);
        assert_eq!(config.initializer, Initializer::Normal(0.0, 1.0));
        let embed: Embedding<TB> = Embedding::new(&config);
        let weights = embed.weight.val().reshape([1000]);
        let (var_act, mean_act) = weights.var_mean(0);
        var_act.to_data().assert_approx_eq(&Data::from([1.0f32]), 1);
        mean_act
            .to_data()
            .assert_approx_eq(&Data::from([0.0f32]), 1);
    }

    #[test]
    fn initializer_zeros() {
        TB::seed(0);
        let config = EmbeddingConfig::new(5, 5).with_initializer(Initializer::Zeros);
        assert_eq!(config.initializer, Initializer::Zeros);
        let conv: Embedding<TB> = Embedding::new(&config);
        for item in conv.weight.to_data().value.iter() {
            assert_eq!(*item, 0.0f32);
        }
    }
}
