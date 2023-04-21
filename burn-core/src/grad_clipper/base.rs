use burn_tensor::{backend::Backend, Tensor, Numeric, Element, TensorKind, Float, BasicOps};
/// Gradient Clipper provides a way to mitigate the exploding gradients
/// problem by clipping every component of the gradient to a value during 
/// backpropagation.
pub struct GradientClipper {
    clip_value: Option<f32>,
    clip_norm: Option<f32>,
}

impl GradientClipper {
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

    fn clip_by_value<B: Backend, const D: usize>(&self, grad: Tensor<B, D>, threshold: f32) -> Tensor<B, D> 
    where
        B::FloatElem: core::cmp::PartialOrd<f32> + From<f32>,
    {
        let mut grad_data = grad.to_data();
        grad_data.value.iter_mut().for_each(|val| {
            if *val > threshold {
                *val = threshold.into();
            } else if *val < -threshold {
                *val = (-threshold).into();
            }
        });

        Tensor::from_data_device(grad_data, &grad.device())
    }

    fn clip_by_norm<B: Backend, const D: usize>(&self, grad: Tensor<B, D>, threshold: f32) -> Tensor<B, D> {
        todo!()
        // Does Tensor have a norm() function?
    }
}