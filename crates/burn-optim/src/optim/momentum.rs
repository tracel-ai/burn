use burn_core as burn;

use burn::config::Config;
use burn::record::Record;
use burn::tensor::backend::Backend;
use burn::tensor::{ElementConversion, Tensor};

/// Configuration to create [momentum](Momentum).
#[derive(Config, Debug)]
pub struct MomentumConfig {
    /// Momemtum factor
    #[config(default = 0.9)]
    pub momentum: f64,
    /// Dampening factor.
    #[config(default = 0.1)]
    pub dampening: f64,
    /// Enables Nesterov momentum, see [On the importance of initialization and
    /// momentum in deep learning](http://www.cs.toronto.edu/~hinton/absps/momentum.pdf).
    #[config(default = false)]
    pub nesterov: bool,
}

/// State of [momentum](Momentum).
#[derive(Record, Clone, new)]
pub struct MomentumState<B: Backend, const D: usize> {
    velocity: Tensor<B, D>,
}

/// Momemtum implementation that transforms gradients.
#[derive(Clone)]
pub struct Momentum<B: Backend> {
    momentum: B::FloatElem,
    dampening: f64,
    nesterov: bool,
}

impl<B: Backend> Momentum<B> {
    /// Creates a new [momentum](Momentum) from a [config](MomentumConfig).
    pub fn new(config: &MomentumConfig) -> Self {
        Self {
            momentum: config.momentum.elem(),
            dampening: config.dampening,
            nesterov: config.nesterov,
        }
    }

    /// Transforms a gradient.
    ///
    /// # Arguments
    ///
    /// * `grad` - Gradient to transform.
    /// * `state` - State of the optimizer.
    ///
    /// # Returns
    ///
    /// * `grad` - Transformed gradient.
    /// * `state` - State of the optimizer.
    pub fn transform<const D: usize>(
        &self,
        grad: Tensor<B, D>,
        state: Option<MomentumState<B, D>>,
    ) -> (Tensor<B, D>, MomentumState<B, D>) {
        let velocity = if let Some(state) = state {
            grad.clone()
                .mul_scalar(1.0 - self.dampening)
                .add(state.velocity.mul_scalar(self.momentum))
        } else {
            grad.clone()
        };

        let grad = match self.nesterov {
            true => velocity.clone().mul_scalar(self.momentum).add(grad),
            false => velocity.clone(),
        };

        (grad, MomentumState::new(velocity))
    }
}

impl<B: Backend, const D: usize> MomentumState<B, D> {
    /// Moves the state to a device.
    ///
    /// # Arguments
    ///
    /// * `device` - Device to move the state to.
    ///
    /// # Returns
    ///
    /// * `self` - Moved state.
    pub fn to_device(mut self, device: &B::Device) -> Self {
        self.velocity = self.velocity.to_device(device);
        self
    }
}
