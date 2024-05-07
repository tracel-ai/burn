use crate as burn;

use crate::tensor::backend::Backend;
use crate::tensor::Tensor;
use crate::{config::Config, module::Module};
use core::marker::PhantomData;

use super::Reduction;

/// Configuration to create a [Huber loss](HuberLoss).
#[derive(Config, Debug)]
pub struct HuberLossConfig {
    /// The bound where the Huber loss function changes from quadratic to linear behaviour.
    pub delta: f32,
}

impl HuberLossConfig {
    /// Initialize [Huber loss](HuberLoss).
    pub fn init<B: Backend>(&self, device: &B::Device) -> HuberLoss<B> {
        // device is not needed as of now, but we might want to prepare some data on it
        // and its consistent with other loss functions
        let _ = device;
        self.assertions();
        HuberLoss {
            delta: self.delta,
            lin_bias: self.delta * self.delta * 0.5,
            _backend: PhantomData,
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
/// L(r) = 0.5 * x^2                  if |r| <= d
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
#[derive(Module, Debug)]
pub struct HuberLoss<B: Backend> {
    delta: f32,
    lin_bias: f32, // delta * delta * 0.5 precomputed
    _backend: PhantomData<B>,
}

impl<B: Backend> HuberLoss<B> {
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
    pub fn forward<const D: usize>(
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
    pub fn forward_no_reduction<const D: usize>(
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
    pub fn forward_residuals<const D: usize>(&self, residuals: Tensor<B, D>) -> Tensor<B, D> {
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

        let inside = residuals.powf_scalar(2.).mul_scalar(0.5);
        inside.mask_where(is_large, outside)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Data;
    use crate::TestBackend;
    type TestTensor<const D: usize> = Tensor<TestBackend, D>;

    #[test]
    fn test_huber_loss() {
        let predict = Data::from([-2., -0.5, 0., 0.3, 1.]);
        let targets = Data::from([0., 0., 0., 0., 0.]);

        let device = Default::default();

        let predict = TestTensor::<1>::from_data(predict, &device);
        let targets = TestTensor::<1>::from_data(targets, &device);

        let huber = HuberLossConfig::new(0.5).init(&device);

        let loss_sum = huber.forward(predict.clone(), targets.clone(), Reduction::Sum);
        let loss = huber.forward(predict.clone(), targets.clone(), Reduction::Auto);
        let loss_no_reduction = huber.forward_no_reduction(predict, targets);

        loss_no_reduction
            .into_data()
            .assert_approx_eq(&Data::from([0.875, 0.125, 0., 0.045, 0.375]), 7);
        loss.into_data().assert_approx_eq(&Data::from([0.284]), 7);
        loss_sum
            .into_data()
            .assert_approx_eq(&Data::from([1.42]), 5);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_huber_ad_loss() {
        type TestAutodiffTensor = Tensor<crate::TestAutodiffBackend, 1>;

        let predict = Data::from([-2., -0.5, 0., 0.3, 1.]);
        let targets = Data::from([0., 0., 0., 0., 0.]);

        let device = Default::default();
        let predict = TestAutodiffTensor::from_data(predict, &device).require_grad();
        let targets = TestAutodiffTensor::from_data(targets, &device);

        let loss = HuberLossConfig::new(0.5).init(&device);
        let loss = loss.forward_no_reduction(predict.clone(), targets);

        let grads = loss.backward();
        let grads_predict = predict.grad(&grads).unwrap();

        grads_predict
            .to_data()
            .assert_approx_eq(&Data::from([-0.5, -0.5, 0., 0.3, 0.5]), 3);
    }
}
