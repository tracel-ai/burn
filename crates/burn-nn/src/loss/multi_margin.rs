use burn_core as burn;

use burn::config::Config;
use burn::module::{Content, DisplaySettings, Module, ModuleDisplay};
use burn::tensor::{Int, Tensor};

use super::Reduction;

/// Configuration to create a [Multi Margin loss](MultiMarginLoss) using the
/// [init function](MultiMarginLossConfig::init).
#[derive(Config, Debug)]
pub struct MultiMarginLossConfig {
    /// The margin between the correct class and the others. Default: `1.0`.
    #[config(default = 1.0)]
    pub margin: f64,
    /// The norm degree for the loss (`1` or `2`). Default: `1`.
    #[config(default = 1)]
    pub p: i32,
}

impl MultiMarginLossConfig {
    /// Initialize [Multi Margin loss](MultiMarginLoss).
    pub fn init(&self) -> MultiMarginLoss {
        self.assertions();
        MultiMarginLoss {
            margin: self.margin,
            p: self.p,
        }
    }

    fn assertions(&self) {
        assert!(
            self.margin >= 0.0,
            "Margin for multi margin loss must be non-negative, got {}",
            self.margin
        );
        assert!(
            self.p == 1 || self.p == 2,
            "Multi margin loss only supports p = 1 or p = 2, got {}",
            self.p
        );
    }
}

/// Multi-class classification margin (hinge) loss between input `x` and target class indices
/// `y`, following
/// [`torch.nn.MultiMarginLoss`](https://pytorch.org/docs/stable/generated/torch.nn.MultiMarginLoss.html).
///
/// For each sample the loss is
///
/// ```text
/// L = (1 / C) * sum over i != y of max(0, margin - x[y] + x[i]) ^ p
/// ```
///
/// where `C` is the number of classes and `y` is the target class index.
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct MultiMarginLoss {
    /// The margin between the correct class and the others.
    pub margin: f64,
    /// The norm degree for the loss (`1` or `2`).
    pub p: i32,
}

impl ModuleDisplay for MultiMarginLoss {
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

impl MultiMarginLoss {
    /// Compute the loss for each sample, then reduce to a single value.
    ///
    /// `Reduction::Auto` behaves as `Reduction::Mean`.
    ///
    /// # Shapes
    ///
    /// - input:  `[batch_size, num_classes]`
    /// - target: `[batch_size]` (class indices in `0..num_classes`)
    /// - output: `[1]`
    pub fn forward(
        &self,
        input: Tensor<2>,
        target: Tensor<1, Int>,
        reduction: Reduction,
    ) -> Tensor<1> {
        let loss = self.forward_no_reduction(input, target);
        match reduction {
            Reduction::Mean | Reduction::Auto => loss.mean(),
            Reduction::Sum => loss.sum(),
            other => panic!("{other:?} reduction is not supported"),
        }
    }

    /// Compute the loss for each sample, without reducing.
    ///
    /// # Shapes
    ///
    /// - input:  `[batch_size, num_classes]`
    /// - target: `[batch_size]` (class indices in `0..num_classes`)
    /// - output: `[batch_size]`
    pub fn forward_no_reduction(&self, input: Tensor<2>, target: Tensor<1, Int>) -> Tensor<1> {
        let [batch_size, num_classes] = input.dims();
        let target_indices = target.reshape([batch_size, 1]);

        // Score of the correct class per sample: [batch_size, 1].
        let correct = input.clone().gather(1, target_indices);

        // Sum over ALL classes of max(0, margin - x[y] + x[i]) ^ p: [batch_size, 1].
        let summed = input
            .sub(correct)
            .add_scalar(self.margin)
            .clamp_min(0.0)
            .powi_scalar(self.p)
            .sum_dim(1);

        // The correct class (i == y) always contributes clamp(margin)^p = margin^p (margin >= 0),
        // so subtract it once to exclude it, then average over the classes.
        // p is restricted to {1, 2}, so compute margin^p without `f64::powi` (std-only).
        let margin_pow_p = if self.p == 2 {
            self.margin * self.margin
        } else {
            self.margin
        };
        let per_sample: Tensor<1> = summed.sub_scalar(margin_pow_p).squeeze_dim(1);
        per_sample.mul_scalar(1.0 / num_classes as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use burn::tensor::Tolerance;
    type FT = f32;

    #[test]
    fn test_multi_margin_loss() {
        let device = Default::default();
        let input = Tensor::<2>::from_data(
            TensorData::from([[0.1, 0.2, 0.7], [0.9, 0.05, 0.05]]),
            &device,
        );
        let target = Tensor::<1, Int>::from_data(TensorData::from([2, 0]), &device);

        let loss = MultiMarginLossConfig::new().init();

        let no_reduction = loss.forward_no_reduction(input.clone(), target.clone());
        let mean = loss.forward(input.clone(), target.clone(), Reduction::Mean);
        let sum = loss.forward(input, target, Reduction::Sum);

        // (1/C) * sum_{i != y} max(0, margin - x[y] + x[i]); margin=1, p=1. Reference from PyTorch.
        let expected = TensorData::from([0.3, 0.1]);
        no_reduction
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());

        mean.into_data()
            .assert_approx_eq::<FT>(&TensorData::from([0.2]), Tolerance::default());
        sum.into_data()
            .assert_approx_eq::<FT>(&TensorData::from([0.4]), Tolerance::default());
    }

    #[test]
    fn test_multi_margin_loss_p2() {
        let device = Default::default();
        let input = Tensor::<2>::from_data(
            TensorData::from([[0.1, 0.2, 0.7], [0.9, 0.05, 0.05]]),
            &device,
        );
        let target = Tensor::<1, Int>::from_data(TensorData::from([2, 0]), &device);

        let loss = MultiMarginLossConfig::new().with_p(2).init();
        let no_reduction = loss.forward_no_reduction(input, target);

        // squared hinge (p = 2); reference from PyTorch.
        let expected = TensorData::from([0.136_667, 0.015]);
        no_reduction
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn display() {
        let config = MultiMarginLossConfig::new().with_margin(0.5);
        let loss = config.init();

        assert_eq!(
            alloc::format!("{loss}"),
            "MultiMarginLoss {margin: 0.5, p: 1}"
        );
    }
}
