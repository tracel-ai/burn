use burn_core as burn;

use crate::RecordState;
use burn::config::Config;
use burn::tensor::Device;
use burn::tensor::Tensor;

/// Configuration to create [weight decay](WeightDecay).
#[derive(Config, Debug)]
pub struct WeightDecayConfig {
    /// The weight decay penalty
    pub penalty: f32,

    /// When true, skip weight decay on tensors with rank < 2.
    /// This applies to biases (rank 1), LayerNorm parameters (rank 1),
    /// and scalars (rank 0). Matches PyTorch's common heuristic.
    #[config(default = false)]
    pub projective: bool,
}

/// State of [weight decay](WeightDecay).
#[derive(RecordState, Clone, new)]
pub struct WeightDecayState<const D: usize> {
    pub(crate) grad_last_step: Tensor<D>,
}

/// Weight decay implementation that transforms gradients.
#[derive(Clone, Debug)]
pub struct WeightDecay {
    penalty: f32,
    projective: bool,
}

impl WeightDecay {
    /// Creates a new [weight decay](WeightDecay) from a [config](WeightDecayConfig).
    pub fn new(config: &WeightDecayConfig) -> Self {
    Self {
        penalty: config.penalty,
        projective: config.projective,
    }
}

    /// Transforms a gradient.
    ///
    /// # Arguments
    ///
    /// * `grad` - Gradient to transform.
    /// * `tensor` - Tensor param of the last iteration.
    ///
    /// # Returns
    ///
    /// * `grad` - Transformed gradient.
   pub fn transform<const D: usize>(&self, grad: Tensor<D>, param: Tensor<D>) -> Tensor<D> {
    let effective_penalty = if self.projective && D < 2 {
        0.0
    } else {
        self.penalty
    };

    if effective_penalty > 0.0 {
        grad + param.mul_scalar(effective_penalty)
    } else {
        grad
    }
}
}

impl<const D: usize> WeightDecayState<D> {
    /// Moves the state to a device.
    ///
    /// # Arguments
    ///
    /// * `device` - Device to move the state to.
    ///
    /// # Returns
    ///
    /// * `self` - Moved state.
    pub fn to_device(mut self, device: &Device) -> Self {
        self.grad_last_step = self.grad_last_step.to_device(device);
        self
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projective_decay_skips_1d_tensors() {
        let device = Device::default();
        let config = WeightDecayConfig::new(0.1).with_projective(true);
        let weight_decay = WeightDecay::new(&config);

        let param = Tensor::<1>::ones([5], &device);
        let grad = Tensor::<1>::ones([5], &device);

        let new_grad = weight_decay.transform(grad.clone(), param.clone());

        let new_data = new_grad.to_data();
        let expected_data = grad.to_data();
        let new_slice = new_data.as_slice::<f32>().unwrap();
        let expected_slice = expected_data.as_slice::<f32>().unwrap();
        for (a, b) in new_slice.iter().zip(expected_slice.iter()) {
            assert!((a - b).abs() < 1e-5, "1D should skip decay: {} vs {}", a, b);
        }
    }

    #[test]
    fn projective_decay_applies_to_2d_tensors() {
        let device = Device::default();
        let config = WeightDecayConfig::new(0.1).with_projective(true);
        let weight_decay = WeightDecay::new(&config);

        let param = Tensor::<2>::ones([2, 3], &device);
        let grad = Tensor::<2>::ones([2, 3], &device);

        let new_grad = weight_decay.transform(grad.clone(), param.clone());
        let expected = grad + param.mul_scalar(0.1);

        let new_data = new_grad.to_data();
        let expected_data = expected.to_data();
        let new_slice = new_data.as_slice::<f32>().unwrap();
        let expected_slice = expected_data.as_slice::<f32>().unwrap();
        for (a, b) in new_slice.iter().zip(expected_slice.iter()) {
            assert!((a - b).abs() < 1e-5, "2D should apply decay: {} vs {}", a, b);
        }
    }

    #[test]
    fn projective_off_does_not_skip_1d_tensors() {
        let device = Device::default();
        let config = WeightDecayConfig::new(0.1).with_projective(false);
        let weight_decay = WeightDecay::new(&config);

        let param = Tensor::<1>::ones([5], &device);
        let grad = Tensor::<1>::ones([5], &device);

        let new_grad = weight_decay.transform(grad.clone(), param.clone());
        let expected = grad + param.mul_scalar(0.1);

        let new_data = new_grad.to_data();
        let expected_data = expected.to_data();
        let new_slice = new_data.as_slice::<f32>().unwrap();
        let expected_slice = expected_data.as_slice::<f32>().unwrap();
        for (a, b) in new_slice.iter().zip(expected_slice.iter()) {
            assert!((a - b).abs() < 1e-5, "1D should decay when off: {} vs {}", a, b);
        }
    }
}