use crate as burn;
use crate::config::Config;
use crate::tensor::Tensor;
use burn_tensor::{backend::Backend, ElementConversion};

/// Gradient Clipper provides a way to mitigate exploding gradients
/// by clipping every component of the gradient to a value during
/// backpropagation.

/// Configuration to create gradient clipper, note that a gradient clipper
/// can be invoked as a standalone struct in a custom training loop that
/// accepts gradients, so you can avoid making it as a config to an optimizer.
#[derive(Config)]
pub struct GradientClipperConfig {
    #[config(default = "Some(1.0)")]
    pub clip_value: Option<f32>,
    #[config(default = "Some(1.0)")]
    pub clip_norm: Option<f32>,
}

pub struct GradientClipper {
    clip_value: Option<f32>,
    clip_norm: Option<f32>,
}

impl GradientClipper {
    pub fn new(config: &GradientClipperConfig) -> Self {
        Self {
            clip_value: config.clip_value,
            clip_norm: config.clip_norm,
        }
    }
    pub fn by_value(threshold: f32) -> Self {
        Self {
            clip_value: Some(threshold),
            clip_norm: None,
        }
    }

    pub fn by_norm(threshold: f32) -> Self {
        Self {
            clip_value: None,
            clip_norm: Some(threshold),
        }
    }

    pub fn clip_gradient<B: Backend, const D: usize>(&self, grad: Tensor<B, D>) -> Tensor<B, D> {
        match (self.clip_value, self.clip_norm) {
            (Some(clip_value), None) => self.clip_by_value(grad, clip_value),
            (None, Some(clip_norm)) => self.clip_by_norm(grad, clip_norm),
            _ => panic!("Gradient Clipper must have one (and only one) clip_value or clip_norm."),
        }
    }

    fn clip_by_value<B: Backend, const D: usize>(
        &self,
        grad: Tensor<B, D>,
        threshold: f32,
    ) -> Tensor<B, D> {
        let mut grad_data = grad.to_data();
        grad_data.value.iter_mut().for_each(|val| {
            let f_val: f32 = val.elem();
            if f_val > threshold {
                *val = threshold.elem();
            } else if f_val < -threshold {
                *val = (-threshold).elem();
            }
        });

        Tensor::from_data_device(grad_data, &grad.device())
    }

    /// Maybe implement on all float tensors? Feel like this could be useful in other contexts.
    fn l2_norm<B: Backend, const D: usize>(tensor: &Tensor<B, D>) -> Tensor<B, 1> {
        let squared = tensor.clone().powf(2.0);
        let sum = squared.sum();

        sum.sqrt()
    }

    fn clip_by_norm<B: Backend, const D: usize>(
        &self,
        grad: Tensor<B, D>,
        threshold: f32,
    ) -> Tensor<B, D> {
        let norm = Self::l2_norm(&grad);
        let norm_float = norm.into_scalar().elem::<f32>();
        if norm_float > threshold {
            let scale = threshold / norm_float;
            grad.mul_scalar(scale)
        } else {
            grad
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    use crate::TestBackend;

    #[test]
    fn test_clip_by_value() {
        let gradient_clipper = GradientClipper::by_value(0.5);

        let gradient: Tensor<TestBackend, 2> = Tensor::from_floats([
            [0.6294, 0.0940, 0.8176, 0.8824, 0.5228, 0.4310],
            [0.7152, 0.9559, 0.7893, 0.5684, 0.5939, 0.8883],
        ]);

        let clipped_gradient = gradient_clipper.clip_gradient(gradient);
        let clipped_gradient_data = clipped_gradient.into_data();

        for value in clipped_gradient_data.value {
            assert!(value <= 0.5);
        }
    }

    #[test]
    fn test_clip_by_norm() {
        let gradient_clipper = GradientClipper::by_norm(2.2);

        let gradient: Tensor<TestBackend, 2> = Tensor::from_floats([
            [0.6294, 0.0940, 0.8176, 0.8824, 0.5228, 0.4310],
            [0.7152, 0.9559, 0.7893, 0.5684, 0.5939, 0.8883],
        ]);

        let clipped_gradient = gradient_clipper.clip_gradient(gradient);
        let clipped_gradient_data = clipped_gradient.into_data();

        for value in clipped_gradient_data.value {
            assert!(value <= 0.88);
        }
    }
}
