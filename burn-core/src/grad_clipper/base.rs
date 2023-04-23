use burn_tensor::{backend::Backend, ElementConversion};
use crate::tensor::Tensor;
/// Gradient Clipper provides a way to mitigate the exploding gradients
/// problem by clipping every component of the gradient to a value during 
/// backpropagation.
pub enum GradientClipper {
    ClipByValue(f32),
    ClipByNorm(f32),
}

impl GradientClipper {

    pub fn clip_gradient<B: Backend, const D: usize>(&self, grad: Tensor<B, D>) -> Tensor<B, D> {
        match self {
            GradientClipper::ClipByValue(threshold) => self.clip_by_value(grad, *threshold),
            GradientClipper::ClipByNorm(max_norm) => self.clip_by_norm(grad, *max_norm),
        }
    }

    fn clip_by_value<B: Backend, const D: usize>(&self, grad: Tensor<B, D>, threshold: f32) -> Tensor<B, D> {
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

    fn clip_by_norm<B: Backend, const D: usize>(&self, grad: Tensor<B, D>, threshold: f32) -> Tensor<B, D> {
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