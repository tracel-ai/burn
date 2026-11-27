use super::{L2Norm, scale_gradient};
use burn_core as burn;

use burn::{
    config::Config,
    tensor::{DType, FloatDType, Tensor},
};

/// Gradient Clipping provides a way to mitigate exploding gradients
#[derive(Config, Debug)]
#[non_exhaustive]
pub enum GradientClippingConfig {
    /// Clip the gradient by value.
    Value(f32),

    /// Limit the L2 norm of each gradient tensor independently.
    Norm(f32),

    /// Limit the joint L2 norm of all parameter gradients in a module optimizer step.
    GlobalNorm(f32),
}

impl GradientClippingConfig {
    /// Initialize the gradient clipping.
    ///
    /// # Returns
    ///
    /// The gradient clipping.
    pub fn init(&self) -> GradientClipping {
        match self {
            GradientClippingConfig::Value(val) => GradientClipping::Value(*val),
            GradientClippingConfig::Norm(val) => GradientClipping::Norm(*val),
            GradientClippingConfig::GlobalNorm(val) => GradientClipping::GlobalNorm(*val),
        }
    }
}

/// Gradient Clipping provides a way to mitigate exploding gradients
/// before an optimizer update.
#[derive(Clone)]
#[non_exhaustive]
pub enum GradientClipping {
    /// Clip the gradient by value.
    Value(f32),

    /// Limit the L2 norm of each gradient tensor independently.
    Norm(f32),

    /// Limit the joint L2 norm of all parameter gradients in a module optimizer step.
    GlobalNorm(f32),
}

impl GradientClipping {
    /// Clip a single gradient. For a single tensor, the global norm equals its own L2 norm.
    ///
    /// # Panics
    ///
    /// Panics if `GlobalNorm` receives a quantized gradient or a negative or NaN limit.
    pub fn clip_gradient<const D: usize>(&self, grad: Tensor<D>) -> Tensor<D> {
        self.validate_global_limit();
        match self {
            GradientClipping::Value(threshold) => self.clip_by_value(grad, *threshold),
            GradientClipping::Norm(max_norm) => self.clip_by_norm(grad, *max_norm),
            GradientClipping::GlobalNorm(max_norm) => {
                let dtype = grad.dtype();
                let coefficient = L2Norm::compute(grad.clone()).clip_coefficient(*max_norm);
                scale_gradient(grad, coefficient, dtype)
            }
        }
    }

    pub(crate) fn validate_global_limit(&self) {
        if let Self::GlobalNorm(limit) = self {
            assert!(
                *limit >= 0.,
                "GlobalNorm requires a nonnegative limit, got {limit}"
            );
        }
    }

    fn clip_by_value<const D: usize>(&self, grad: Tensor<D>, threshold: f32) -> Tensor<D> {
        let greater_mask = grad.clone().greater_scalar(threshold);
        let lower_mask = grad.clone().lower_scalar(-threshold);

        let clipped_grad = grad.mask_fill(greater_mask, threshold);

        clipped_grad.mask_fill(lower_mask, -threshold)
    }

    fn clip_by_norm<const D: usize>(&self, grad: Tensor<D>, threshold: f32) -> Tensor<D> {
        // Compute the norm in F32: in F16 the sum of squares overflows once the norm
        // exceeds ~256, and BF16 loses too much precision accumulating it.
        let dtype = grad.dtype();
        let grad = match dtype {
            DType::F16 | DType::BF16 => grad.cast(FloatDType::F32),
            _ => grad,
        };

        let norm = Self::l2_norm(grad.clone());
        let min_positive = grad
            .dtype()
            .finfo()
            .unwrap_or(FloatDType::F32.finfo())
            .min_positive;
        let clip_coef = threshold / norm.add_scalar(min_positive);
        let clip_coef_clamped = clip_coef.clamp_max(1.0);
        grad.mul(clip_coef_clamped.unsqueeze()).cast(dtype)
    }

    fn l2_norm<const D: usize>(tensor: Tensor<D>) -> Tensor<1> {
        let squared = tensor.square();
        let sum = squared.sum();
        sum.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Tensor;

    #[test]
    fn global_norm_clips_tiny_finite_gradients() {
        let grad = Tensor::<1>::from_floats([3e-30, 4e-30], &Default::default());
        let scaled = GradientClipping::GlobalNorm(1e-30).clip_gradient(grad) * 1e30;
        let data = scaled.into_data();
        data.assert_approx_eq::<f32>(
            &burn::tensor::TensorData::from([0.6f32, 0.8]),
            Default::default(),
        );
    }

    #[test]
    fn global_norm_preserves_small_components_when_clipping() {
        let large = 2.0_f32.powi(60);
        let small = 2.0_f32.powi(-100);
        let grad = Tensor::<1>::from_data([large, small], (&Default::default(), DType::F32));
        let clipped = GradientClipping::GlobalNorm(large / 2.).clip_gradient(grad);
        clipped.into_data().assert_eq(
            &burn::tensor::TensorData::from([large / 2., small / 2.]),
            true,
        );
    }

    #[test]
    fn global_norm_rounds_subnormal_gradients_once() {
        let x = f32::from_bits(1);
        let grad = Tensor::<1>::from_data([x; 9], (&Default::default(), DType::F32));
        let clipped = GradientClipping::GlobalNorm(x).clip_gradient(grad);
        clipped
            .into_data()
            .assert_eq(&burn::tensor::TensorData::from([0f32; 9]), true);
    }

    #[test]
    fn global_norm_infinite_limit_preserves_infinite_and_finite_values() {
        let grad = Tensor::<1>::from_floats([f32::INFINITY, 3.], &Default::default());
        let clipped = GradientClipping::GlobalNorm(f32::INFINITY).clip_gradient(grad.clone());
        clipped.into_data().assert_eq(&grad.into_data(), true);
    }

    #[test]
    #[should_panic(expected = "GlobalNorm requires a nonnegative limit")]
    fn global_norm_rejects_negative_single_tensor_limit() {
        let grad = Tensor::<1>::ones([1], &Default::default());
        let _ = GradientClipping::GlobalNorm(-1.).clip_gradient(grad);
    }

    #[test]
    fn test_clip_by_value() {
        let gradient: Tensor<2> = Tensor::from_floats(
            [
                [0.6294, 0.0940, 0.8176, 0.8824, 0.5228, 0.4310],
                [0.7152, 0.9559, 0.7893, 0.5684, 0.5939, 0.8883],
            ],
            &Default::default(),
        );

        let clipped_gradient = GradientClipping::Value(0.5).clip_gradient(gradient);
        let clipped_gradient_data = clipped_gradient.into_data();

        for value in clipped_gradient_data.iter::<f32>() {
            assert!(value <= 0.5);
        }
    }

    #[test]
    fn test_clip_by_norm() {
        let gradient: Tensor<2> = Tensor::from_floats(
            [
                [0.6294, 0.0940, 0.8176, 0.8824, 0.5228, 0.4310],
                [0.7152, 0.9559, 0.7893, 0.5684, 0.5939, 0.8883],
            ],
            &Default::default(),
        );

        let clipped_gradient = GradientClipping::Norm(2.2).clip_gradient(gradient);
        let clipped_gradient_data = clipped_gradient.into_data();

        for value in clipped_gradient_data.iter::<f32>() {
            assert!(value <= 0.88);
        }
    }
    #[test]
    fn test_clip_by_norm_no_clipping() {
        let gradient: Tensor<2> = Tensor::from_floats(
            [[0.3, 0.4, 0.5, 0.2], [0.1, 0.6, 0.3, 0.4]],
            &Default::default(),
        );

        let clipped_gradient = GradientClipping::Norm(2.2).clip_gradient(gradient.clone());

        clipped_gradient
            .into_data()
            .assert_eq(&gradient.into_data(), true);
    }

    #[test]
    fn test_clip_by_norm_f16_does_not_overflow() {
        let gradient = Tensor::<1>::from_floats([300.0], &Default::default()).cast(FloatDType::F16);

        let clipped_gradient = GradientClipping::Norm(1.0).clip_gradient(gradient);

        assert_eq!(clipped_gradient.dtype(), DType::F16);
        let actual = clipped_gradient.into_scalar::<f32>();
        assert!((actual - 1.0).abs() < 1e-3, "actual={actual}");
    }

    #[test]
    fn test_clip_by_norm_f16_norm_above_f16_max() {
        let gradient =
            Tensor::<1>::from_floats([60000.0, 60000.0], &Default::default()).cast(FloatDType::F16);

        let clipped_gradient = GradientClipping::Norm(1.0).clip_gradient(gradient);

        let expected = core::f32::consts::FRAC_1_SQRT_2;
        for value in clipped_gradient.into_data().iter::<f32>() {
            assert!((value - expected).abs() < 1e-3, "value={value}");
        }
    }
}
