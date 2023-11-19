use crate as burn;

use super::Initializer;
use crate::config::Config;
use crate::module::Module;
use crate::module::Param;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;
use burn_tensor::Int;

/// Configuration to create an [Embedding](Embedding) layer.
#[derive(Config)]
pub struct EmbeddingConfig {
    /// The number of embedding vectors.
    n_embedding: usize,
    /// The size of each vector.
    d_model: usize,
    /// The type of function used to initialize neural network parameters
    #[config(default = "Initializer::Normal{mean:0.0, std:1.0}")]
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

impl EmbeddingConfig {
    /// Initialize a new [embedding](Embedding) module.
    pub fn init<B: Backend>(&self) -> Embedding<B> {
        let weight = self
            .initializer
            .init([self.n_embedding, self.d_model])
            .require_grad();

        Embedding {
            weight: Param::from(weight),
        }
    }
    /// Initialize a new [embedding](Embedding) module with a [record](EmbeddingRecord).
    pub fn init_with<B: Backend>(&self, record: EmbeddingRecord<B>) -> Embedding<B> {
        Embedding {
            weight: record.weight,
        }
    }
}

impl<B: Backend> Embedding<B> {
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
    use super::*;
    use crate::TestBackend;
    use burn_tensor::Data;

    #[test]
    fn initializer_default() {
        TestBackend::seed(0);

        let config = EmbeddingConfig::new(100, 10);
        let embed = config.init::<TestBackend>();
        let weights = embed.weight.val().reshape([1000]);
        let (var_act, mean_act) = weights.var_mean(0);

        assert_eq!(
            config.initializer,
            Initializer::Normal {
                mean: 0.0,
                std: 1.0
            }
        );
        var_act.to_data().assert_approx_eq(&Data::from([1.0f32]), 0);
        mean_act
            .to_data()
            .assert_approx_eq(&Data::from([0.0f32]), 0);
    }

    #[test]
    fn initializer_zeros() {
        TestBackend::seed(0);

        let config = EmbeddingConfig::new(5, 5).with_initializer(Initializer::Zeros);
        let embed = config.init::<TestBackend>();

        assert_eq!(config.initializer, Initializer::Zeros);
        embed
            .weight
            .to_data()
            .assert_approx_eq(&Data::zeros(embed.weight.shape()), 3);
    }
}
