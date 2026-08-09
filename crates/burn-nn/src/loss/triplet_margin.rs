use burn_core as burn;

use burn::config::Config;
use burn::module::{Content, DisplaySettings, Module, ModuleDisplay};
use burn::tensor::Tensor;
use burn::tensor::linalg::lp_norm;

use super::Reduction;

/// Configuration to create a [Triplet Margin loss](TripletMarginLoss) using the
/// [init function](TripletMarginLossConfig::init).
#[derive(Config, Debug)]
pub struct TripletMarginLossConfig {
    /// The margin between the positive and negative distances. Default: `1.0`.
    #[config(default = 1.0)]
    pub margin: f64,
    /// The degree of the norm used for the pairwise distance. Default: `2.0`.
    #[config(default = 2.0)]
    pub p: f64,
}

impl TripletMarginLossConfig {
    /// Initialize [Triplet Margin loss](TripletMarginLoss).
    pub fn init(&self) -> TripletMarginLoss {
        TripletMarginLoss {
            margin: self.margin,
            p: self.p,
        }
    }
}

/// Measures the triplet loss given an anchor, a positive, and a negative input, following
/// [`torch.nn.TripletMarginLoss`](https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html).
///
/// For each triplet the loss is
///
/// ```text
/// L = max(0, d(anchor, positive) - d(anchor, negative) + margin)
/// ```
///
/// where `d(x, y) = ||x - y||_p` is the pairwise distance. It encourages the anchor to be
/// closer to the positive than to the negative by at least `margin`.
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct TripletMarginLoss {
    /// The margin between the positive and negative distances.
    pub margin: f64,
    /// The degree of the norm used for the pairwise distance.
    pub p: f64,
}

impl ModuleDisplay for TripletMarginLoss {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("margin", &self.margin)
            .add("p", &self.p)
            .optional()
    }
}

impl TripletMarginLoss {
    /// Compute the loss for each triplet, then reduce to a single value.
    ///
    /// `Reduction::Auto` behaves as `Reduction::Mean`.
    ///
    /// # Shapes
    ///
    /// - anchor:   `[batch_size, embedding_dim]`
    /// - positive: `[batch_size, embedding_dim]`
    /// - negative: `[batch_size, embedding_dim]`
    /// - output:   `[1]`
    pub fn forward(
        &self,
        anchor: Tensor<2>,
        positive: Tensor<2>,
        negative: Tensor<2>,
        reduction: Reduction,
    ) -> Tensor<1> {
        let loss = self.forward_no_reduction(anchor, positive, negative);
        match reduction {
            Reduction::Mean | Reduction::Auto => loss.mean(),
            Reduction::Sum => loss.sum(),
            other => panic!("{other:?} reduction is not supported"),
        }
    }

    /// Compute the loss for each triplet, without reducing.
    ///
    /// # Shapes
    ///
    /// - anchor:   `[batch_size, embedding_dim]`
    /// - positive: `[batch_size, embedding_dim]`
    /// - negative: `[batch_size, embedding_dim]`
    /// - output:   `[batch_size]`
    pub fn forward_no_reduction(
        &self,
        anchor: Tensor<2>,
        positive: Tensor<2>,
        negative: Tensor<2>,
    ) -> Tensor<1> {
        // Pairwise distances over the embedding dim: shape [batch_size, 1] -> [batch_size].
        let distance_positive: Tensor<1> =
            lp_norm(anchor.clone() - positive, self.p, 1).squeeze_dim(1);
        let distance_negative: Tensor<1> = lp_norm(anchor - negative, self.p, 1).squeeze_dim(1);

        // max(0, d_positive - d_negative + margin)
        (distance_positive - distance_negative)
            .add_scalar(self.margin)
            .clamp_min(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use burn::tensor::Tolerance;
    type FT = f32;

    #[test]
    fn test_triplet_margin_loss() {
        let device = Default::default();
        // Sample 0: normal; sample 1: partial; sample 2: clamped to zero.
        let anchor = Tensor::<2>::from_data(
            TensorData::from([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]),
            &device,
        );
        let positive = Tensor::<2>::from_data(
            TensorData::from([[1.0, 1.0], [1.0, 2.0], [0.0, 0.0]]),
            &device,
        );
        let negative = Tensor::<2>::from_data(
            TensorData::from([[1.0, 0.0], [0.0, 0.0], [5.0, 0.0]]),
            &device,
        );

        let loss = TripletMarginLossConfig::new().init();

        let no_reduction =
            loss.forward_no_reduction(anchor.clone(), positive.clone(), negative.clone());
        let mean = loss.forward(
            anchor.clone(),
            positive.clone(),
            negative.clone(),
            Reduction::Mean,
        );
        let sum = loss.forward(anchor, positive, negative, Reduction::Sum);

        // max(0, ||a-p||_2 - ||a-n||_2 + margin); margin defaults to 1.0. Reference from PyTorch.
        let expected = TensorData::from([1.414214, 0.585786, 0.0]);
        no_reduction
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());

        mean.into_data()
            .assert_approx_eq::<FT>(&TensorData::from([0.666_667]), Tolerance::default());
        sum.into_data()
            .assert_approx_eq::<FT>(&TensorData::from([2.0]), Tolerance::default());
    }

    #[test]
    fn display() {
        let config = TripletMarginLossConfig::new().with_margin(0.5);
        let loss = config.init();

        assert_eq!(
            alloc::format!("{loss}"),
            "TripletMarginLoss {margin: 0.5, p: 2}"
        );
    }
}
