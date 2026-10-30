use crate::Initializer;
use burn_core as burn;

use alloc::vec::Vec;
use burn::config::Config;
use burn::instrument::{self, Bounds, compute};
use burn::module::Module;
use burn::module::Param;
use burn::module::{Content, DisplaySettings, ModuleDisplay};
use burn::tensor::Int;
use burn::tensor::{DType, Device, Tensor};

use burn::tensor::module::embedding;

/// Configuration to create an [Embedding](Embedding) layer using the [init function](EmbeddingConfig::init).
#[derive(Config, Debug)]
pub struct EmbeddingConfig {
    /// The number of embedding vectors.
    pub n_embedding: usize,
    /// The size of each vector.
    pub d_model: usize,
    /// The type of function used to initialize neural network parameters
    #[config(default = "Initializer::Normal{mean:0.0, std:1.0}")]
    pub initializer: Initializer,
}

/// Lookup table to store a fix number of vectors.
///
/// Should be created with [EmbeddingConfig].
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct Embedding {
    /// The learnable weights of the module of shape `[n_embedding, d_model]` initialized
    /// from a normal distribution `N(0, 1)`.
    pub weight: Param<Tensor<2>>,
}

impl ModuleDisplay for Embedding {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        let [n_embedding, d_model] = self.weight.shape().dims();
        content
            .add("n_embedding", &n_embedding)
            .add("d_model", &d_model)
            .optional()
    }
}

impl EmbeddingConfig {
    /// Initialize a new [embedding](Embedding) module.
    pub fn init(&self, device: &Device) -> Embedding {
        let weight = self
            .initializer
            .init([self.n_embedding, self.d_model], device);

        Embedding { weight }
    }
}

impl Embedding {
    /// The floor for a lookup alone: read the indices and each distinct row they select once,
    /// write the embeddings once.
    ///
    /// Reads the indices back to count the distinct rows, so it is evaluated only for an
    /// observer that prices.
    ///
    /// None for a quantized table, whose rows come out at a width the table does not say.
    fn bounds(&self, input: &Tensor<2, Int>) -> Option<Bounds> {
        let weight = self.weight.val();
        let width = match weight.dtype() {
            DType::QFloat(_) => return None,
            dtype => dtype.size() as u64,
        };
        let [_, d_model] = weight.dims().map(|dim| dim as u64);
        let [batch_size, seq_length] = input.dims().map(|dim| dim as u64);
        let mut indices: Vec<i64> = input.clone().into_data().iter::<i64>().collect();
        indices.sort_unstable();
        indices.dedup();
        let row = d_model * width;
        Some(Bounds {
            read: instrument::tensor_bytes(input) + indices.len() as u64 * row,
            write: batch_size * seq_length * row,
            flops: 0,
        })
    }

    /// Applies the forward pass on the input tensor.
    ///
    /// See also [embedding](burn::tensor::module::embedding).
    ///
    /// # Shapes
    ///
    /// - input: `[batch_size, seq_length]`
    /// - output: `[batch_size, seq_length, d_model]`
    #[compute(bounds = Self::bounds)]
    pub fn forward(&self, input: Tensor<2, Int>) -> Tensor<3> {
        embedding(self.weight.val(), input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use burn::tensor::Tolerance;
    type FT = f32;

    /// A lookup reads each distinct row it selects once, however often it repeats.
    #[cfg(feature = "std")]
    #[test]
    fn a_lookup_reads_each_distinct_row_once() {
        let device = Default::default();
        let embedding = EmbeddingConfig::new(100, 8).init(&device);
        let input = Tensor::<2, Int>::from_ints([[1, 2, 1], [2, 2, 5]], &device);
        let index = input.dtype().size() as u64;

        let recorded = crate::recording::recorded(|| {
            embedding.forward(input);
        });

        assert_eq!(recorded.lines(), ["embedding"]);
        assert_eq!(
            recorded.declared("embedding"),
            Some(Bounds {
                read: 2 * 3 * index + 3 * 8 * 4,
                write: 2 * 3 * 8 * 4,
                flops: 0,
            })
        );
    }

    #[test]
    fn initializer_zeros() {
        let device = Device::default();
        device.seed(0);

        let config = EmbeddingConfig::new(5, 5).with_initializer(Initializer::Zeros);
        let embed = config.init(&Default::default());

        assert_eq!(config.initializer, Initializer::Zeros);
        embed.weight.to_data().assert_approx_eq::<FT>(
            &TensorData::zeros::<f32, _>(embed.weight.shape()),
            Tolerance::default(),
        );
    }

    #[test]
    fn display() {
        let config = EmbeddingConfig::new(100, 10);
        let embed = config.init(&Default::default());

        assert_eq!(
            alloc::format!("{embed}"),
            "Embedding {n_embedding: 100, d_model: 10, params: 1000}"
        );
    }
}
