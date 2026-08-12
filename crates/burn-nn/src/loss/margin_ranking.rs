use burn_core as burn;

use burn::module::{Content, DisplaySettings, ModuleDisplay};
use burn::tensor::Tensor;
use burn::tensor::activation::relu;
use burn::{config::Config, module::Module};

use super::Reduction;

/// Configuration to create a [Margin Ranking loss](MarginRankingLoss).
#[derive(Config, Debug)]
pub struct MarginRankingLossConfig {
    /// The margin by which the correctly-ranked pair should be separated. Default: `0.0`.
    #[config(default = 0.0)]
    pub margin: f64,
}

impl MarginRankingLossConfig {
    /// Initialize [Margin Ranking loss](MarginRankingLoss).
    pub fn init(&self) -> MarginRankingLoss {
        self.assertions();
        MarginRankingLoss {
            margin: self.margin,
        }
    }

    fn assertions(&self) {
        assert!(
            self.margin >= 0.0,
            "Margin for margin ranking loss must be a non-negative number."
        );
    }
}

/// Measures the loss for ranking two inputs, following
/// [`torch.nn.MarginRankingLoss`](https://pytorch.org/docs/stable/generated/torch.nn.MarginRankingLoss.html).
///
/// Given first inputs `x1`, second inputs `x2`, and a target `y` (each element `+1` or
/// `-1`), the loss for each element is
///
/// ```text
/// L = max(0, -y * (x1 - x2) + margin)
/// ```
///
/// A target of `+1` means `x1` should be ranked higher than `x2`, and `-1` the reverse.
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct MarginRankingLoss {
    /// The margin by which the correctly-ranked pair should be separated.
    pub margin: f64,
}

impl ModuleDisplay for MarginRankingLoss {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content.add("margin", &self.margin).optional()
    }
}

impl MarginRankingLoss {
    /// Compute the loss element-wise for the inputs and target, then reduce to a
    /// single loss value.
    ///
    /// `Reduction::Auto` behaves as `Reduction::Mean`.
    ///
    /// # Shapes
    ///
    /// - first: \[...dims\]
    /// - second: \[...dims\]
    /// - target: \[...dims\]
    /// - output: \[1\]
    pub fn forward<const D: usize>(
        &self,
        first: Tensor<D>,
        second: Tensor<D>,
        target: Tensor<D>,
        reduction: Reduction,
    ) -> Tensor<1> {
        let loss = self.forward_no_reduction(first, second, target);
        match reduction {
            Reduction::Mean | Reduction::Auto => loss.mean(),
            Reduction::Sum => loss.sum(),
            other => panic!("{other:?} reduction is not supported"),
        }
    }

    /// Compute the loss element-wise for the inputs and target.
    ///
    /// # Shapes
    ///
    /// - first: [...dims]
    /// - second: [...dims]
    /// - target: [...dims]
    /// - output: [...dims]
    pub fn forward_no_reduction<const D: usize>(
        &self,
        first: Tensor<D>,
        second: Tensor<D>,
        target: Tensor<D>,
    ) -> Tensor<D> {
        // -y * (x1 - x2) + margin, then clamp negatives to zero via relu.
        let scaled = target.mul(first - second).neg().add_scalar(self.margin);
        relu(scaled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use burn::tensor::Tolerance;
    type FT = f32;

    #[test]
    fn test_margin_ranking_loss() {
        let device = Default::default();

        let first = Tensor::<1>::from_data(TensorData::from([1., 2., 3.]), &device);
        let second = Tensor::<1>::from_data(TensorData::from([2., 1., 0.5]), &device);
        let target = Tensor::<1>::from_data(TensorData::from([1., -1., 1.]), &device);

        let loss = MarginRankingLossConfig::new().with_margin(0.5).init();

        let no_reduction = loss.forward_no_reduction(first.clone(), second.clone(), target.clone());
        let mean = loss.forward(
            first.clone(),
            second.clone(),
            target.clone(),
            Reduction::Mean,
        );
        let sum = loss.forward(first, second, target, Reduction::Sum);

        let expected = TensorData::from([1.5, 1.5, 0.0]);
        no_reduction
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());

        let expected = TensorData::from([1.0]);
        mean.into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());

        let expected = TensorData::from([3.0]);
        sum.into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn display() {
        let config = MarginRankingLossConfig::new().with_margin(0.5);
        let loss = config.init();

        assert_eq!(alloc::format!("{loss}"), "MarginRankingLoss {margin: 0.5}");
    }
}
