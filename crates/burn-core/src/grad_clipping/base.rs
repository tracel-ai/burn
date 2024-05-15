use crate as burn;

use crate::{config::Config, tensor::Tensor};
use burn_tensor::backend::Backend;

/// Gradient Clipping provides a way to mitigate exploding gradients
#[derive(Config)]
pub enum GradientClippingConfig {
    /// Clip the gradient by value.
    Value(f32),

    /// Clip the gradient by norm.
    Norm(f32),
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
        }
    }
}

/// Gradient Clipping provides a way to mitigate exploding gradients
/// by clipping every component of the gradient by value or by norm during
/// backpropagation.
#[derive(Clone)]
pub enum GradientClipping {
    /// Clip the gradient by value.
    Value(f32),

    /// Clip the gradient by norm.
    Norm(f32),
}

impl GradientClipping {
    /// Clip the gradient.
    ///
    /// # Arguments
    ///
    /// * `grad` - The gradient to clip.
    ///
    /// # Returns
    ///
    /// The clipped gradient.
    pub fn clip_gradient<B: Backend, const D: usize>(&self, grad: Tensor<B, D>) -> Tensor<B, D> {
        match self {
            GradientClipping::Value(threshold) => self.clip_by_value(grad, *threshold),
            GradientClipping::Norm(max_norm) => self.clip_by_norm(grad, *max_norm),
        }
    }

    fn clip_by_value<B: Backend, const D: usize>(
        &self,
        grad: Tensor<B, D>,
        threshold: f32,
    ) -> Tensor<B, D> {
        let greater_mask = grad.clone().greater_elem(threshold);
        let lower_mask = grad.clone().lower_elem(-threshold);

        let clipped_grad = grad.mask_fill(greater_mask, threshold);

        clipped_grad.mask_fill(lower_mask, -threshold)
    }

    #[cfg(all(not(feature = "wasm-sync"), target_family = "wasm"))]
    fn clip_by_norm<B: Backend, const D: usize>(
        &self,
        _grad: Tensor<B, D>,
        _threshold: f32,
    ) -> Tensor<B, D> {
        todo!("Not yet supported on wasm");
    }

    #[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
    fn clip_by_norm<B: Backend, const D: usize>(
        &self,
        grad: Tensor<B, D>,
        threshold: f32,
    ) -> Tensor<B, D> {
        use burn_tensor::ElementConversion;

        let norm = Self::l2_norm(grad.clone());
        let norm_float = norm.into_scalar().elem::<f32>();

        if norm_float > threshold {
            let scale = threshold / norm_float;
            grad.mul_scalar(scale)
        } else {
            grad
        }
    }

    #[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
    fn l2_norm<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Tensor<B, 1> {
        let squared = tensor.powf_scalar(2.0);
        let sum = squared.sum();

        sum.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    use crate::TestBackend;

    #[test]
    fn test_clip_by_value() {
        let gradient: Tensor<TestBackend, 2> = Tensor::from_floats(
            [
                [0.6294, 0.0940, 0.8176, 0.8824, 0.5228, 0.4310],
                [0.7152, 0.9559, 0.7893, 0.5684, 0.5939, 0.8883],
            ],
            &Default::default(),
        );

        let clipped_gradient = GradientClipping::Value(0.5).clip_gradient(gradient);
        let clipped_gradient_data = clipped_gradient.into_data();

        for value in clipped_gradient_data.value {
            assert!(value <= 0.5);
        }
    }

    #[test]
    fn test_clip_by_norm() {
        let gradient: Tensor<TestBackend, 2> = Tensor::from_floats(
            [
                [0.6294, 0.0940, 0.8176, 0.8824, 0.5228, 0.4310],
                [0.7152, 0.9559, 0.7893, 0.5684, 0.5939, 0.8883],
            ],
            &Default::default(),
        );

        let clipped_gradient = GradientClipping::Norm(2.2).clip_gradient(gradient);
        let clipped_gradient_data = clipped_gradient.into_data();

        for value in clipped_gradient_data.value {
            assert!(value <= 0.88);
        }
    }
}
