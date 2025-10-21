use burn_core as burn;

use burn::module::{Content, DisplaySettings, ModuleDisplay};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn::{config::Config, module::Module};

use super::Reduction;

/// Configuration to create a [Huber loss](HuberLoss).
#[derive(Config, Debug)]
pub struct HuberLossConfig {
    /// The bound where the Huber loss function changes from quadratic to linear behaviour.
    pub delta: f32,
}

impl HuberLossConfig {
    /// Initialize [Huber loss](HuberLoss).
    pub fn init(&self) -> HuberLoss {
        self.assertions();
        HuberLoss {
            delta: self.delta,
            lin_bias: self.delta * self.delta * 0.5,
        }
    }

    fn assertions(&self) {
        assert!(
            self.delta >= 0., // This also tests for normality
            "Delta for Huber loss must be a non-negative number."
        );
    }
}

/// Calculate the Huber loss between the inputs and the target.
///
/// The loss for each element of the residuals `r = targets - predictions` is given by
///
/// ```text
/// L(r) = 0.5 * r^2                  if |r| <= d
/// L(r) = 0.5 * d^2 + d * (|r| - d)  if |r| >  d
/// ```
///
/// where `d` is the configured `delta`. In particular, this is equal to the
/// [L2 Loss](super::MseLoss) for residuals with magnitude smaller than `delta`,
/// but behaves linearly instead of quadratically for large residuals.
///
/// This loss function is less sensitive to outliers than the mean squared error loss.
///
/// See also: <https://en.wikipedia.org/wiki/Huber_loss>
#[derive(Module, Debug, Clone)]
#[module(custom_display)]
pub struct HuberLoss {
    /// The bound where the Huber loss function changes from quadratic to linear behaviour.
    pub delta: f32,
    /// Precomputed value for the linear bias.
    pub lin_bias: f32, // delta * delta * 0.5 precomputed
}

impl ModuleDisplay for HuberLoss {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("delta", &self.delta)
            .add("lin_bias", &self.lin_bias)
            .optional()
    }
}

impl HuberLoss {
    /// Compute the loss element-wise for the predictions and targets, then reduce
    /// to a single loss value.
    ///
    /// `Reduction::Auto` behaves as `Reduction::Mean`.
    ///
    /// # Shapes
    ///
    /// - predictions: \[...dims\]
    /// - targets: \[...dims\]
    /// - output: \[1\]
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
    /// Compute the loss element-wise for the predictions and targets.
    ///
    /// # Shapes
    ///
    /// - predictions: [...dims]
    /// - targets: [...dims]
    /// - output: [...dims]
    pub fn forward_no_reduction<const D: usize, B: Backend>(
        &self,
        predictions: Tensor<B, D>,
        targets: Tensor<B, D>,
    ) -> Tensor<B, D> {
        let residuals = targets - predictions;
        self.forward_residuals(residuals)
    }
    /// Compute the loss element-wise for the given residuals.
    ///
    /// # Shapes
    ///
    /// - residuals: [...dims]
    /// - output: [...dims]
    pub fn forward_residuals<const D: usize, B: Backend>(
        &self,
        residuals: Tensor<B, D>,
    ) -> Tensor<B, D> {
        let is_large = residuals.clone().abs().greater_elem(self.delta);
        // We are interested in `sign(r)` when `abs(r) > self.delta`. Note that the
        // `sign()` function, in general, suffers from a jump at 0.
        // Instead the following tensor implements `delta * sign(r)` for values outside
        // the bound:
        let softsign = residuals.clone().clamp(-self.delta, self.delta);

        // 0.5 * d^2 + d * (|r| - d) =
        // d * |r| - 0.5 * d^2
        // Moreover |r| = sign(r) * r
        let outside = softsign.mul(residuals.clone()).sub_scalar(self.lin_bias);

        let inside = residuals.square().mul_scalar(0.5);
        inside.mask_where(is_large, outside)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn::tensor::TensorData;
    type TestTensor<const D: usize> = Tensor<TestBackend, D>;
    use burn::tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    #[test]
    fn test_huber_loss() {
        let predict = TensorData::from([-2., -0.5, 0., 0.3, 1.]);
        let targets = TensorData::from([0., 0., 0., 0., 0.]);

        let device = Default::default();

        let predict = TestTensor::<1>::from_data(predict, &device);
        let targets = TestTensor::<1>::from_data(targets, &device);

        let huber = HuberLossConfig::new(0.5).init();

        let loss_sum = huber.forward(predict.clone(), targets.clone(), Reduction::Sum);
        let loss = huber.forward(predict.clone(), targets.clone(), Reduction::Auto);
        let loss_no_reduction = huber.forward_no_reduction(predict, targets);

        let expected = TensorData::from([0.875, 0.125, 0., 0.045, 0.375]);
        loss_no_reduction
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());

        let expected = TensorData::from([0.284]);
        loss.into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());

        let expected = TensorData::from([1.42]);
        loss_sum
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_huber_ad_loss() {
        type TestAutodiffTensor = Tensor<crate::TestAutodiffBackend, 1>;

        let predict = TensorData::from([-2., -0.5, 0., 0.3, 1.]);
        let targets = TensorData::from([0., 0., 0., 0., 0.]);

        let device = Default::default();
        let predict = TestAutodiffTensor::from_data(predict, &device).require_grad();
        let targets = TestAutodiffTensor::from_data(targets, &device);

        let loss = HuberLossConfig::new(0.5).init();
        let loss = loss.forward_no_reduction(predict.clone(), targets);

        let grads = loss.backward();
        let grads_predict = predict.grad(&grads).unwrap();

        let expected = TensorData::from([-0.5, -0.5, 0., 0.3, 0.5]);
        grads_predict
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn display() {
        let config = HuberLossConfig::new(0.5);
        let loss = config.init();

        assert_eq!(
            alloc::format!("{loss}"),
            "HuberLoss {delta: 0.5, lin_bias: 0.125}"
        );
    }
}
