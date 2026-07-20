use burn_core as burn;

use burn::config::Config;
use burn::module::{Content, DisplaySettings, Module, ModuleDisplay};
use burn::tensor::Tensor;
use burn::tensor::linalg::lp_norm;

/// Configuration to create a [PairwiseDistance](PairwiseDistance) layer using the
/// [init function](PairwiseDistanceConfig::init).
#[derive(Config, Debug)]
pub struct PairwiseDistanceConfig {
    /// The norm degree for the pairwise distance. Default: `2.0`.
    #[config(default = 2.0)]
    pub p: f64,
    /// Small value added to the difference for numerical stability. Default: `1e-6`.
    #[config(default = 1e-6)]
    pub eps: f64,
}

impl PairwiseDistanceConfig {
    /// Initialize a new [PairwiseDistance](PairwiseDistance) layer.
    pub fn init(&self) -> PairwiseDistance {
        PairwiseDistance {
            p: self.p,
            eps: self.eps,
        }
    }
}

/// Computes the pairwise distance between the rows of two tensors, following
/// [`torch.nn.PairwiseDistance`](https://pytorch.org/docs/stable/generated/torch.nn.PairwiseDistance.html).
///
/// For each row, the distance is `||x1 - x2 + eps||_p` over the last (feature) dimension, which is
/// reduced in the output (equivalent to `keepdim = false`).
///
/// Should be created with [PairwiseDistanceConfig].
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct PairwiseDistance {
    /// The norm degree for the pairwise distance.
    pub p: f64,
    /// Small value added to the difference for numerical stability.
    pub eps: f64,
}

impl ModuleDisplay for PairwiseDistance {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content.add("p", &self.p).add("eps", &self.eps).optional()
    }
}

impl PairwiseDistance {
    /// Applies the forward pass, computing one distance per row.
    ///
    /// # Shapes
    ///
    /// - x1:     `[batch_size, features]`
    /// - x2:     `[batch_size, features]`
    /// - output: `[batch_size]`
    pub fn forward(&self, x1: Tensor<2>, x2: Tensor<2>) -> Tensor<1> {
        // ||x1 - x2 + eps||_p over the feature dimension: [batch_size, 1] -> [batch_size].
        let diff = (x1 - x2).add_scalar(self.eps);
        lp_norm(diff, self.p, 1).squeeze_dim(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use burn::tensor::Tolerance;
    type FT = f32;

    #[test]
    fn pairwise_distance_p2() {
        let device = Default::default();
        let x1 = Tensor::<2>::from_data(TensorData::from([[1.0, 2.0], [3.0, 4.0]]), &device);
        let x2 = Tensor::<2>::from_data(TensorData::from([[1.0, 1.0], [0.0, 0.0]]), &device);

        let output = PairwiseDistanceConfig::new().init().forward(x1, x2);

        // ||x1 - x2 + eps||_2 per row; reference from PyTorch (eps = 1e-6 is negligible here).
        let expected = TensorData::from([1.0, 5.0]);
        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn pairwise_distance_p1() {
        let device = Default::default();
        let x1 = Tensor::<2>::from_data(TensorData::from([[1.0, 2.0], [3.0, 4.0]]), &device);
        let x2 = Tensor::<2>::from_data(TensorData::from([[1.0, 1.0], [0.0, 0.0]]), &device);

        let output = PairwiseDistanceConfig::new()
            .with_p(1.0)
            .init()
            .forward(x1, x2);

        // L1 distance per row; reference from PyTorch.
        let expected = TensorData::from([1.0, 7.0]);
        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn display() {
        let layer = PairwiseDistanceConfig::new()
            .with_p(2.0)
            .with_eps(0.5)
            .init();

        assert_eq!(
            alloc::format!("{layer}"),
            "PairwiseDistance {p: 2, eps: 0.5}"
        );
    }
}
