use burn_core as burn;

use burn::config::Config;
use burn::module::{Content, DisplaySettings, Module, ModuleDisplay};
use burn::tensor::Tensor;

use super::Reduction;

/// Configuration to create a [Hinge Embedding loss](HingeEmbeddingLoss) using the
/// [init function](HingeEmbeddingLossConfig::init).
#[derive(Config, Debug)]
pub struct HingeEmbeddingLossConfig {
    /// The margin applied to negative pairs (target `-1`). Default: `1.0`.
    #[config(default = 1.0)]
    pub margin: f64,
}

impl HingeEmbeddingLossConfig {
    /// Initialize [Hinge Embedding loss](HingeEmbeddingLoss).
    pub fn init(&self) -> HingeEmbeddingLoss {
        HingeEmbeddingLoss {
            margin: self.margin,
        }
    }
}

/// Measures the loss given an input tensor `x` and a target tensor `y` (each element `1` or
/// `-1`), following
/// [`torch.nn.HingeEmbeddingLoss`](https://pytorch.org/docs/stable/generated/torch.nn.HingeEmbeddingLoss.html).
///
/// The loss for each element is
///
/// ```text
/// L = x                    if y == 1
///     max(0, margin - x)   if y == -1
/// ```
///
/// This is typically used for learning nonlinear embeddings or for semi-supervised learning,
/// where `y == 1` marks similar pairs and `y == -1` marks dissimilar pairs.
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct HingeEmbeddingLoss {
    /// The margin applied to negative pairs (target `-1`).
    pub margin: f64,
}

impl ModuleDisplay for HingeEmbeddingLoss {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content.add("margin", &self.margin).optional()
    }
}

impl HingeEmbeddingLoss {
    /// Compute the loss element-wise for the input and target, then reduce to a single value.
    ///
    /// `Reduction::Auto` behaves as `Reduction::Mean`.
    ///
    /// # Shapes
    ///
    /// - input:  `[...dims]`
    /// - target: `[...dims]` (values in `{-1, 1}`)
    /// - output: `[1]`
    pub fn forward<const D: usize>(
        &self,
        input: Tensor<D>,
        target: Tensor<D>,
        reduction: Reduction,
    ) -> Tensor<1> {
        let loss = self.forward_no_reduction(input, target);
        match reduction {
            Reduction::Mean | Reduction::Auto => loss.mean(),
            Reduction::Sum => loss.sum(),
            other => panic!("{other:?} reduction is not supported"),
        }
    }

    /// Compute the loss element-wise for the input and target.
    ///
    /// # Shapes
    ///
    /// - input:  `[...dims]`
    /// - target: `[...dims]` (values in `{-1, 1}`)
    /// - output: `[...dims]`
    pub fn forward_no_reduction<const D: usize>(
        &self,
        input: Tensor<D>,
        target: Tensor<D>,
    ) -> Tensor<D> {
        // y == 1  -> x ;  y == -1 -> max(0, margin - x)
        let negative = input.clone().neg().add_scalar(self.margin).clamp_min(0.0);
        let positive_mask = target.equal_scalar(1);
        negative.mask_where(positive_mask, input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use burn::tensor::Tolerance;
    type FT = f32;

    #[test]
    fn test_hinge_embedding_loss() {
        let device = Default::default();
        let input = Tensor::<1>::from_data(TensorData::from([0.5, 2.0, 1.5]), &device);
        let target = Tensor::<1>::from_data(TensorData::from([1.0, -1.0, -1.0]), &device);

        let loss = HingeEmbeddingLossConfig::new().init();

        let no_reduction = loss.forward_no_reduction(input.clone(), target.clone());
        let mean = loss.forward(input.clone(), target.clone(), Reduction::Mean);
        let sum = loss.forward(input, target, Reduction::Sum);

        // x if y == 1 else max(0, margin - x); margin defaults to 1.0. Reference from PyTorch.
        let expected = TensorData::from([0.5, 0.0, 0.0]);
        no_reduction
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());

        mean.into_data()
            .assert_approx_eq::<FT>(&TensorData::from([0.166_667]), Tolerance::default());
        sum.into_data()
            .assert_approx_eq::<FT>(&TensorData::from([0.5]), Tolerance::default());
    }

    #[test]
    fn display() {
        let config = HingeEmbeddingLossConfig::new().with_margin(0.5);
        let loss = config.init();

        assert_eq!(alloc::format!("{loss}"), "HingeEmbeddingLoss {margin: 0.5}");
    }
}
