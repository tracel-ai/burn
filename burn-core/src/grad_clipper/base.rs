use burn_tensor::{backend::Backend, Tensor};

/// Gradient Clipper provides a way to mitigate the exploding gradients
/// problem by clipping every component of the gradient to a value during 
/// backpropagation.
pub struct GradientClipper {
    clip_value: Option<f64>,
    clip_norm: Option<f64>,
}

impl GradientClipper {
    pub fn by_value(threshold: f64) -> Self {
        Self {
            clip_value: Some(threshold),
            clip_norm: None,
        }
    }

    pub fn by_norm(threshold: f64) -> Self {
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

    fn clip_by_value<B: Backend, const D: usize>(&self, grad: Tensor<B, D>, threshold: f64) -> Tensor<B, D> {
        todo!()
    }

    fn clip_by_norm<B: Backend, const D: usize>(&self, grad: Tensor<B, D>, threshold: f64) -> Tensor<B, D> {
        todo!()
    }
}