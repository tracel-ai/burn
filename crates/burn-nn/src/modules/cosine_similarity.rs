use burn_core as burn;

use burn::config::Config;
use burn::module::{Content, DisplaySettings, Module, ModuleDisplay};
use burn::tensor::linalg::cosine_similarity;
use burn::tensor::{AsIndex, Tensor};

/// Configuration to create a [CosineSimilarity](CosineSimilarity) layer using the
/// [init function](CosineSimilarityConfig::init).
#[derive(Config, Debug)]
pub struct CosineSimilarityConfig {
    /// The dimension along which the cosine similarity is computed. Default: `1`.
    /// Negative dimensions are supported and count from the end.
    #[config(default = 1)]
    pub dim: isize,
    /// Small value to avoid division by zero. Default: `1e-8`.
    #[config(default = 1e-8)]
    pub eps: f64,
}

impl CosineSimilarityConfig {
    /// Initialize a new [CosineSimilarity](CosineSimilarity) layer.
    pub fn init(&self) -> CosineSimilarity {
        CosineSimilarity {
            dim: self.dim,
            eps: self.eps,
        }
    }
}

/// Computes the cosine similarity between two tensors along a dimension, following
/// [`torch.nn.CosineSimilarity`](https://pytorch.org/docs/stable/generated/torch.nn.CosineSimilarity.html).
///
/// The similarity is the dot product of the inputs divided by the product of their L2 norms
/// (clamped by `eps`). The chosen dimension is reduced in the output.
///
/// Should be created with [CosineSimilarityConfig].
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct CosineSimilarity {
    /// The dimension along which the cosine similarity is computed.
    /// Negative dimensions are supported and count from the end.
    pub dim: isize,
    /// Small value to avoid division by zero.
    pub eps: f64,
}

impl ModuleDisplay for CosineSimilarity {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("dim", &self.dim)
            .add("eps", &self.eps)
            .optional()
    }
}

impl CosineSimilarity {
    /// Applies the forward pass, computing one similarity per row.
    ///
    /// # Shapes
    ///
    /// - x1:     `[batch_size, features]`
    /// - x2:     `[batch_size, features]`
    /// - output: `[batch_size]`
    pub fn forward(&self, x1: Tensor<2>, x2: Tensor<2>) -> Tensor<1> {
        let dim = self.dim.expect_dim_index(2);
        cosine_similarity(x1, x2, dim, Some(self.eps)).squeeze_dim(dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use burn::tensor::Tolerance;
    type FT = f32;

    #[test]
    fn cosine_similarity_dim1() {
        let device = Default::default();
        let x1 = Tensor::<2>::from_data(TensorData::from([[1.0, 0.0], [1.0, 1.0]]), &device);
        let x2 = Tensor::<2>::from_data(TensorData::from([[1.0, 0.0], [0.0, 1.0]]), &device);

        let output = CosineSimilarityConfig::new().init().forward(x1, x2);

        // dot(x1, x2) / (||x1|| * ||x2||) per row; reference from PyTorch.
        let expected = TensorData::from([1.0, 0.707107]);
        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn cosine_similarity_negative_dim() {
        let device = Default::default();
        let x1 = Tensor::<2>::from_data(TensorData::from([[1.0, 0.0], [1.0, 1.0]]), &device);
        let x2 = Tensor::<2>::from_data(TensorData::from([[1.0, 0.0], [0.0, 1.0]]), &device);

        let output = CosineSimilarityConfig::new()
            .with_dim(-1)
            .init()
            .forward(x1, x2);

        let expected = TensorData::from([1.0, 0.707107]);
        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn display() {
        let layer = CosineSimilarityConfig::new()
            .with_dim(-1)
            .with_eps(0.5)
            .init();

        assert_eq!(
            alloc::format!("{layer}"),
            "CosineSimilarity {dim: -1, eps: 0.5}"
        );
    }
}
