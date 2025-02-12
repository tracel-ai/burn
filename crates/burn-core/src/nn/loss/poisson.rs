use core::f32::consts::PI;

use crate as burn;
use crate::module::{Content, DisplaySettings, ModuleDisplay};
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;
use crate::{config::Config, module::Module};

use super::Reduction;

/// Configuration for creating a [PoissonNllLoss](PoissonNllLoss) instance.
///
/// This configuration allows customization of the Poisson Negative Log Likelihood (NLL) loss
/// behavior, such as whether the input is in log-space, whether to include the Stirling
/// approximation term, and a small epsilon value to avoid numerical instability.
#[derive(Config, Debug)]
pub struct PoissonNllLossConfig {
    /// If `true`, the predictions are expected to be in log-space.
    ///
    /// When `log_input` is `true`, the loss is computed as:
    /// ```text
    /// L(predictions, target) = exp(predictions) - target * predictions
    /// ```
    /// When `log_input` is `false`, the loss is computed as:
    /// ```text
    /// L(predictions, target) = predictions - target * log(predictions + eps)
    /// ```
    #[config(default = true)]
    pub log_input: bool,
    /// Whether to compute the full loss, including the Stirling approximation term.
    ///
    /// When `full` is `true`, the Stirling approximation term is added to the loss:
    /// ```text
    /// target * log(target) - target + 0.5 * log(2 * PI * target)
    /// ```
    #[config(default = false)]
    pub full: bool,
    /// A small value to avoid evaluation of `log(0)` when `log_input` is `false`.
    ///
    /// This epsilon value is added to the predictions to ensure numerical stability
    /// when computing the logarithm.
    #[config(default = 1e-8)]
    pub eps: f64,
}

impl PoissonNllLossConfig {
    /// Initializes a [PoissonNllLoss](PoissonNllLoss) instance with the current configuration.
    ///
    /// # Panics
    /// - Panics if `eps` is not a positive number.
    pub fn init(&self) -> PoissonNllLoss {
        self.assertions();
        PoissonNllLoss {
            log_input: self.log_input,
            full: self.full,
            eps: self.eps,
        }
    }

    /// Validates the configuration parameters.
    ///
    /// # Panics
    /// - Panics if `eps` is not a positive number.
    fn assertions(&self) {
        assert!(
            self.eps > 0.,
            "eps for PoissonNllLoss must be a positive number."
        );
    }
}

/// Negative Log Likelihood (NLL) loss with a Poisson distribution assumption for the target.
///
/// This loss function is used when the target values are assumed to follow a Poisson distribution.
/// The loss is defined as:
/// ```text
/// target ~ Poisson(input)
/// L(predictions, target) = predictions - target * log(predictions) + log(target!)
/// ```
/// The last term (`log(target!)`) can be omitted or approximated using Stirling's formula.
/// The approximation is applied for `target > 1`, while for `target <= 1`, zeros are added to the loss.
///
/// For more details, see:
/// <https://en.wikipedia.org/wiki/Poisson_regression#Maximum_likelihood-based_parameter_estimation>
#[derive(Module, Debug, Clone)]
#[module(custom_display)]
pub struct PoissonNllLoss {
    /// If `true`, the predictions are expected to be in log-space.
    pub log_input: bool,
    /// Whether to compute the full loss, including the Stirling approximation term.
    pub full: bool,
    /// A small value to avoid evaluation of `log(0)` when `log_input` is `false`.
    pub eps: f64,
}

impl ModuleDisplay for PoissonNllLoss {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("log_input", &self.log_input)
            .add("full", &self.full)
            .add("eps", &self.eps)
            .optional()
    }
}

impl PoissonNllLoss {
    /// Computes the loss element-wise for the given predictions and targets, then reduces
    /// the result to a single loss value.
    ///
    /// # Arguments
    /// - `predictions`: The predicted values.
    /// - `targets`: The target values.
    /// - `reduction`: The reduction method to apply. `Reduction::Auto` behaves as `Reduction::Mean`.
    ///
    /// # Shapes
    /// - `predictions`: `[...dims]`
    /// - `targets`: `[...dims]`
    /// - `output`: `[1]`
    ///
    /// # Panics
    /// - Panics if the shapes of `predictions` and `targets` do not match.
    /// - Panics if any target value is negative.
    /// - Panics if `log_input` is `false` and any prediction value is negative.
    pub fn forward<const D: usize, B: Backend>(
        &self,
        predictions: Tensor<B, D>,
        targets: Tensor<B, D>,
        reduction: Reduction,
    ) -> Tensor<B, 1> {
        let loss = self.forward_no_reduction(predictions, targets);
        match reduction {
            Reduction::Mean | Reduction::Auto => loss.mean(),
            Reduction::Sum => loss.sum(),
        }
    }

    /// Computes the loss element-wise for the given predictions and targets without reduction.
    ///
    /// # Arguments
    /// - `predictions`: The predicted values.
    /// - `targets`: The target values.
    ///
    /// # Shapes
    /// - `predictions`: `[...dims]`
    /// - `targets`: `[...dims]`
    /// - `output`: `[...dims]`
    ///
    /// # Panics
    /// - Panics if the shapes of `predictions` and `targets` do not match.
    /// - Panics if any target value is negative.
    /// - Panics if `log_input` is `false` and any prediction value is negative.
    pub fn forward_no_reduction<const D: usize, B: Backend>(
        &self,
        predictions: Tensor<B, D>,
        targets: Tensor<B, D>,
    ) -> Tensor<B, D> {
        self.assertions(&predictions, &targets);
        let mut loss;
        if self.log_input {
            loss = predictions.clone().exp() - targets.clone() * predictions;
        } else {
            loss = predictions.clone() - targets.clone() * (predictions + self.eps).log();
        }
        if self.full {
            let log_stirling_term = targets.clone() * targets.clone().log() - targets.clone()
                + (targets.clone() * 2. * PI).log() * 0.5;
            loss = loss
                + log_stirling_term
                    .mask_where(targets.clone().lower_equal_elem(1), targets.zeros_like());
        }
        loss
    }

    /// Validates the input tensors for the loss computation.
    ///
    /// # Panics
    /// - Panics if the shapes of `predictions` and `targets` do not match.
    /// - Panics if any target value is negative.
    /// - Panics if `log_input` is `false` and any prediction value is negative.
    fn assertions<const D: usize, B: Backend>(
        &self,
        predictions: &Tensor<B, D>,
        targets: &Tensor<B, D>,
    ) {
        let predictions_dims = predictions.dims();
        let targets_dims = targets.dims();
        assert!(
            predictions_dims == targets_dims,
            "Shape of targets ({:?}) should correspond to outer shape of predictions ({:?}).",
            targets_dims,
            predictions_dims
        );
        assert!(
            targets.clone().greater_equal_elem(0.).all().into_scalar(),
            "All the values of `targets` must be non-negative."
        );
        if !self.log_input {
            assert!(
                predictions.clone().greater_equal_elem(0.).all().into_scalar(),
                "When `log_input` is `false`, all the values of `predictions` must be non-negative."
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorData;
    use crate::TestBackend;
    type TestTensor<const D: usize> = Tensor<TestBackend, D>;

    #[test]
    fn test_poisson_nll_loss() {
        let predictions = TensorData::from([0., 0., -40., 1., 2., 3.]);
        let targets = TensorData::from([1., 4.5, 2.5, 0., 0., 2.]);

        let device = Default::default();

        let predictions = TestTensor::<1>::from_data(predictions, &device);
        let targets = TestTensor::<1>::from_data(targets, &device);

        let poisson = PoissonNllLossConfig::new().init();

        let loss_sum = poisson.forward(predictions.clone(), targets.clone(), Reduction::Sum);
        let loss = poisson.forward(predictions.clone(), targets.clone(), Reduction::Auto);
        let loss_no_reduction = poisson.forward_no_reduction(predictions, targets);

        let expected = TensorData::from([1.0000, 1.0000, 100.0000, 2.7183, 7.3891, 14.0855]);
        loss_no_reduction.into_data().assert_approx_eq(&expected, 5);

        let expected = TensorData::from([21.0321]);
        loss.into_data().assert_approx_eq(&expected, 5);

        let expected = TensorData::from([126.1929]);
        loss_sum.into_data().assert_approx_eq(&expected, 5);
    }

    #[test]
    fn test_poisson_nll_loss_no_log_input() {
        let predictions = TensorData::from([0.0, 0.5, 1.0, 1.0, 2.71828, 7.38905, 20.0855]);
        let targets = TensorData::from([2., 3., 1., 4.5, 0., 0., 2.]);

        let device = Default::default();

        let predictions = TestTensor::<1>::from_data(predictions, &device);
        let targets = TestTensor::<1>::from_data(targets, &device);

        let poisson = PoissonNllLossConfig::new().with_log_input(false).init();

        let loss_no_reduction = poisson.forward_no_reduction(predictions.clone(), targets.clone());

        let expected = TensorData::from([36.84136, 2.579441, 1.0, 1.0, 2.71828, 7.38905, 14.0855]);
        loss_no_reduction.into_data().assert_approx_eq(&expected, 5);
    }

    #[test]
    fn test_poisson_nll_loss_full() {
        let predictions = TensorData::from([0., 0., -40., 1., 2., 3.]);
        let targets = TensorData::from([1., 4.5, 2.5, 0., 0., 2.]);

        let device = Default::default();

        let predictions = TestTensor::<1>::from_data(predictions, &device);
        let targets = TestTensor::<1>::from_data(targets, &device);

        let poisson = PoissonNllLossConfig::new().with_full(true).init();

        let loss_sum = poisson.forward(predictions.clone(), targets.clone(), Reduction::Sum);
        let loss = poisson.forward(predictions.clone(), targets.clone(), Reduction::Auto);
        let loss_no_reduction = poisson.forward_no_reduction(predictions, targets);

        let expected = TensorData::from([1.0000, 4.9393, 101.1678, 2.7183, 7.3891, 14.7373]);
        loss_no_reduction.into_data().assert_approx_eq(&expected, 5);

        let expected = TensorData::from([21.9920]);
        loss.into_data().assert_approx_eq(&expected, 5);

        let expected = TensorData::from([131.9518]);
        loss_sum.into_data().assert_approx_eq(&expected, 5);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_poisson_nll_loss_gradients() {
        type TestAutodiffTensor = Tensor<crate::TestAutodiffBackend, 1>;

        let predictions = TensorData::from([0., 0., -40., 1., 2., 3.]);
        let targets = TensorData::from([1., 4.5, 2.5, 0., 0., 2.]);

        let device = Default::default();

        let predictions1 = TestAutodiffTensor::from_data(predictions, &device).require_grad();
        let predictions2 = predictions1.clone();
        let targets = TestAutodiffTensor::from_data(targets, &device);

        let poisson = PoissonNllLossConfig::new().with_full(false).init();
        let poisson_full = PoissonNllLossConfig::new().with_full(true).init();

        let loss_sum = poisson.forward(predictions1.clone(), targets.clone(), Reduction::Sum);
        let loss_full_sum =
            poisson_full.forward(predictions2.clone(), targets.clone(), Reduction::Sum);

        let grads = loss_sum.backward();
        let grads_full = loss_full_sum.backward();

        let grads_predictions1 = predictions1.grad(&grads).unwrap();
        let grads_predictions2 = predictions2.grad(&grads_full).unwrap();

        let expected = TensorData::from([0.0000, -3.5000, -2.5000, 2.7183, 7.3891, 18.0855]);

        grads_predictions1
            .into_data()
            .assert_approx_eq(&expected, 5);
        grads_predictions2
            .into_data()
            .assert_approx_eq(&expected, 5);
    }

    #[test]
    #[should_panic = "eps for PoissonNllLoss must be a positive number."]
    fn test_negative_eps() {
        let _poisson = PoissonNllLossConfig::new().with_eps(0.).init();
    }

    #[test]
    #[should_panic = "All the values of `targets` must be non-negative."]
    fn test_targets_with_negative_values() {
        let predictions = TensorData::from([0., 0., -40., 1., 2., 3., 4.]);
        let targets = TensorData::from([1., 4.5, 2.5, 0., 0., 2., -0.42]);

        let device = Default::default();

        let predictions = TestTensor::<1>::from_data(predictions, &device);
        let targets = TestTensor::<1>::from_data(targets, &device);

        let poisson = PoissonNllLossConfig::new().init();

        let _loss = poisson.forward(predictions.clone(), targets.clone(), Reduction::Auto);
    }

    #[test]
    #[should_panic = "Shape of targets"]
    fn test_shape_tensors() {
        let predictions = TensorData::from([0., 1., 2.]);
        let targets = TensorData::from([0., 1.]);

        let device = Default::default();

        let predictions = TestTensor::<1>::from_data(predictions, &device);
        let targets = TestTensor::<1>::from_data(targets, &device);

        let poisson = PoissonNllLossConfig::new().init();

        let _loss = poisson.forward_no_reduction(predictions.clone(), targets.clone());
    }

    #[test]
    #[should_panic = "When `log_input` is `false`, all the values of `predictions` must be non-negative."]
    fn test_exp_predictions_non_negative() {
        let predictions = TensorData::from([0.3, -0.1, 0.4]);
        let targets = TensorData::from([0., 1., 0.]);

        let device = Default::default();

        let predictions = TestTensor::<1>::from_data(predictions, &device);
        let targets = TestTensor::<1>::from_data(targets, &device);

        let poisson = PoissonNllLossConfig::new().with_log_input(false).init();

        let _loss = poisson.forward_no_reduction(predictions.clone(), targets.clone());
    }

    #[test]
    fn display() {
        let config = PoissonNllLossConfig::new();
        let loss = config.init();

        assert_eq!(
            alloc::format!("{}", loss),
            "PoissonNllLoss {log_input: true, full: false, eps: 0.00000001}"
        );
    }
}
