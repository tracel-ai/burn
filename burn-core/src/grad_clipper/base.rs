use crate as burn;
use crate::config::Config;
use crate::tensor::Tensor;
use burn_tensor::{backend::Backend, ElementConversion};

/// Gradient Clipper provides a way to mitigate exploding gradients
/// by clipping every component of the gradient to a value during
/// backpropagation.
#[derive(Config)]
pub struct GradientClipper {
    #[config(default = "Some(1.0)")] // ngl this is confusing
    pub clip_value: Option<f32>,
    #[config(default = "Some(1.0)")]
    pub clip_norm: Option<f32>,
}

impl GradientClipper {
    pub fn new() -> Self {
        Self {
            clip_value: None,
            clip_norm: None,
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
        let norm = sum.sqrt();
        norm
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
            let scaled_grad = grad.mul_scalar(scale);
            scaled_grad
        } else {
            grad
        }
    }
}
