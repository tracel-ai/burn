use burn_tensor::backend::Backend;

use crate as burn;
use crate::record::Record;

use crate::config::Config;
use crate::tensor::{ElementConversion, Tensor};

/// Configuration to create [WeightDecay](WeightDecay).
#[derive(Config)]
pub struct WeightDecayConfig {
    /// L2 penalty.
    pub penalty: f64,
}

#[derive(Record, Clone, new)]
pub struct WeightDecayState<B: Backend, const D: usize> {
    grad_last_step: Tensor<B, D>,
}

/// Weight decay implementation that transforms gradients.
pub struct WeightDecay<B: Backend> {
    penalty: B::FloatElem,
}

impl<B: Backend> WeightDecay<B> {
    pub fn new(config: &WeightDecayConfig) -> Self {
        Self {
            penalty: config.penalty.elem(),
        }
    }

    pub fn transform<const D: usize>(
        &self,
        grad: Tensor<B, D>,
        state: Option<WeightDecayState<B, D>>,
    ) -> (Tensor<B, D>, WeightDecayState<B, D>) {
        let grad_last_step = grad.clone();

        let grad = match state {
            Some(state) => state.grad_last_step.mul_scalar(self.penalty).add(grad),
            None => grad,
        };

        (grad, WeightDecayState::new(grad_last_step))
    }
}

impl<B: Backend, const D: usize> WeightDecayState<B, D> {
    pub fn to_device(mut self, device: &B::Device) -> Self {
        self.grad_last_step = self.grad_last_step.to_device(device);
        self
    }
}
