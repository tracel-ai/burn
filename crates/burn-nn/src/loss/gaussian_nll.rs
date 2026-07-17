use burn_core as burn;

use burn::config::Config;
use burn::module::{Content, DisplaySettings, Module, ModuleDisplay};
use burn::tensor::Tensor;

use super::Reduction;

/// Constant `0.5 * ln(2 * pi)`, added when `full` is enabled. Hardcoded to keep this `no_std`.
const HALF_LN_TWO_PI: f64 = 0.918_938_533_204_672_74;

/// Configuration to create a [Gaussian NLL loss](GaussianNLLLoss) using the
/// [init function](GaussianNLLLossConfig::init).
#[derive(Config, Debug)]
pub struct GaussianNLLLossConfig {
    /// Value used to clamp `var` for stability. Default: `1e-6`.
    #[config(default = 1e-6)]
    pub eps: f64,
    /// Whether to include the constant term `0.5 * ln(2 * pi)`. Default: `false`.
    #[config(default = false)]
    pub full: bool,
}

impl GaussianNLLLossConfig {
    /// Initialize [Gaussian NLL loss](GaussianNLLLoss).
    pub fn init(&self) -> GaussianNLLLoss {
        GaussianNLLLoss {
            eps: self.eps,
            full: self.full,
        }
    }
}

/// Negative log likelihood loss for a Gaussian target with predicted mean and variance,
/// following
/// [`torch.nn.GaussianNLLLoss`](https://pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html).
///
/// For each element, with `var` clamped to at least `eps`, the loss is
///
/// ```text
/// L = 0.5 * (log(var) + (input - target)^2 / var)   (+ 0.5 * log(2 * pi) if `full`)
/// ```
///
/// `input` is the predicted mean and `var` the predicted (positive) variance.
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct GaussianNLLLoss {
    /// Value used to clamp `var` for stability.
    pub eps: f64,
    /// Whether to include the constant term `0.5 * ln(2 * pi)`.
    pub full: bool,
}

impl ModuleDisplay for GaussianNLLLoss {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("eps", &self.eps)
            .add("full", &self.full)
            .optional()
    }
}

impl GaussianNLLLoss {
    /// Compute the loss element-wise, then reduce to a single value.
    ///
    /// `Reduction::Auto` behaves as `Reduction::Mean`.
    ///
    /// # Shapes
    ///
    /// - input:  `[...dims]` (predicted mean)
    /// - target: `[...dims]`
    /// - var:    `[...dims]` (predicted variance, positive)
    /// - output: `[1]`
    pub fn forward<const D: usize>(
        &self,
        input: Tensor<D>,
        target: Tensor<D>,
        var: Tensor<D>,
        reduction: Reduction,
    ) -> Tensor<1> {
        let loss = self.forward_no_reduction(input, target, var);
        match reduction {
            Reduction::Mean | Reduction::Auto => loss.mean(),
            Reduction::Sum => loss.sum(),
            other => panic!("{other:?} reduction is not supported"),
        }
    }

    /// Compute the loss element-wise.
    ///
    /// # Shapes
    ///
    /// - input:  `[...dims]` (predicted mean)
    /// - target: `[...dims]`
    /// - var:    `[...dims]` (predicted variance, positive)
    /// - output: `[...dims]`
    pub fn forward_no_reduction<const D: usize>(
        &self,
        input: Tensor<D>,
        target: Tensor<D>,
        var: Tensor<D>,
    ) -> Tensor<D> {
        // Clamp the variance for numerical stability.
        let var = var.clamp_min(self.eps);

        // 0.5 * (log(var) + (input - target)^2 / var)
        let squared_error = (input - target).square();
        let loss = var
            .clone()
            .log()
            .add(squared_error.div(var))
            .mul_scalar(0.5);

        if self.full {
            loss.add_scalar(HALF_LN_TWO_PI)
        } else {
            loss
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use burn::tensor::Tolerance;
    type FT = f32;

    #[test]
    fn test_gaussian_nll_loss() {
        let device = Default::default();
        let input = Tensor::<1>::from_data(TensorData::from([1.0, 2.0]), &device);
        let target = Tensor::<1>::from_data(TensorData::from([1.5, 1.0]), &device);
        let var = Tensor::<1>::from_data(TensorData::from([0.5, 1.0]), &device);

        let loss = GaussianNLLLossConfig::new().init();

        let no_reduction = loss.forward_no_reduction(input.clone(), target.clone(), var.clone());
        let mean = loss.forward(input.clone(), target.clone(), var.clone(), Reduction::Mean);
        let sum = loss.forward(input, target, var, Reduction::Sum);

        // 0.5 * (log(var) + (input - target)^2 / var); reference from PyTorch.
        let expected = TensorData::from([-0.096574, 0.5]);
        no_reduction
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());

        mean.into_data()
            .assert_approx_eq::<FT>(&TensorData::from([0.201713]), Tolerance::default());
        sum.into_data()
            .assert_approx_eq::<FT>(&TensorData::from([0.403426]), Tolerance::default());
    }

    #[test]
    fn test_gaussian_nll_loss_full() {
        let device = Default::default();
        let input = Tensor::<1>::from_data(TensorData::from([1.0, 2.0]), &device);
        let target = Tensor::<1>::from_data(TensorData::from([1.5, 1.0]), &device);
        let var = Tensor::<1>::from_data(TensorData::from([0.5, 1.0]), &device);

        let loss = GaussianNLLLossConfig::new().with_full(true).init();
        let no_reduction = loss.forward_no_reduction(input, target, var);

        // base + 0.5 * ln(2 * pi); reference from PyTorch.
        let expected = TensorData::from([0.822365, 1.418939]);
        no_reduction
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn display() {
        let config = GaussianNLLLossConfig::new().with_eps(0.5);
        let loss = config.init();

        assert_eq!(
            alloc::format!("{loss}"),
            "GaussianNLLLoss {eps: 0.5, full: false}"
        );
    }
}
