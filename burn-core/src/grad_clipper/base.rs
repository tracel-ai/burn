use crate::tensor::Tensor;
use burn_tensor::{backend::Backend, ElementConversion};

/// Gradient Clipper provides a way to mitigate exploding gradients
/// by clipping every component of the gradient to a value during
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

    fn clip_by_value<B: Backend, const D: usize>(
        &self,
        grad: Tensor<B, D>,
        threshold: f32,
    ) -> Tensor<B, D> {
        let greater_mask = B::greater_elem(grad.clone().into_primitive(), threshold.elem());
        let lower_mask = B::lower_elem(grad.clone().into_primitive(), (-threshold).elem());

        let clipped_grad = B::mask_fill(grad.into_primitive(), greater_mask, threshold.elem());
        let clipped_grad = B::mask_fill(clipped_grad, lower_mask, (-threshold).elem());

        Tensor::from_primitive(clipped_grad)
    }

    fn l2_norm<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Tensor<B, 1> {
        let squared = tensor.powf(2.0);
        let sum = squared.sum();

        sum.sqrt()
    }

    fn clip_by_norm<B: Backend, const D: usize>(
        &self,
        grad: Tensor<B, D>,
        threshold: f32,
    ) -> Tensor<B, D> {
        let norm = Self::l2_norm(grad.clone());
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
        let gradient: Tensor<TestBackend, 2> = Tensor::from_floats([
            [0.6294, 0.0940, 0.8176, 0.8824, 0.5228, 0.4310],
            [0.7152, 0.9559, 0.7893, 0.5684, 0.5939, 0.8883],
        ]);

        let clipped_gradient = GradientClipper::ClipByValue(0.5).clip_gradient(gradient);
        let clipped_gradient_data = clipped_gradient.into_data();

        for value in clipped_gradient_data.value {
            assert!(value <= 0.5);
        }
    }

    #[test]
    fn test_clip_by_norm() {
        let gradient: Tensor<TestBackend, 2> = Tensor::from_floats([
            [0.6294, 0.0940, 0.8176, 0.8824, 0.5228, 0.4310],
            [0.7152, 0.9559, 0.7893, 0.5684, 0.5939, 0.8883],
        ]);

        let clipped_gradient = GradientClipper::ClipByNorm(2.2).clip_gradient(gradient);
        let clipped_gradient_data = clipped_gradient.into_data();

        for value in clipped_gradient_data.value {
            assert!(value <= 0.88);
        }
    }
}
