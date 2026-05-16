use burn_core as burn;

use burn::config::Config;
use burn::record::Record;
use burn::tensor::Device;
use burn::tensor::Tensor;

/// Configuration to create [weight decay](WeightDecay).
#[derive(Config, Debug)]
pub struct WeightDecayConfig {
    /// L2 penalty.
    pub penalty: f32,
}

/// State of [weight decay](WeightDecay).
#[derive(Record, Clone, new)]
pub struct WeightDecayState<const D: usize> {
    pub(crate) grad_last_step: Tensor<D>,
}

/// Weight decay implementation that transforms gradients.
#[derive(Clone)]
pub struct WeightDecay {
    penalty: f32,
}

impl WeightDecay {
    /// Creates a new [weight decay](WeightDecay) from a [config](WeightDecayConfig).
    pub fn new(config: &WeightDecayConfig) -> Self {
        Self {
            penalty: config.penalty,
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
    pub fn transform<const D: usize>(&self, grad: Tensor<D>, tensor: Tensor<D>) -> Tensor<D> {
        tensor.mul_scalar(self.penalty).add(grad)
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
