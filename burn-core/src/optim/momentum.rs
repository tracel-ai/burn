use crate as burn;

use super::visitor::GradientsParams;
use super::{load_state_gradients, register_state_gradients};
use crate::config::Config;
use crate::module::{ParamId, StateNamed};
use crate::tensor::backend::ADBackend;
use crate::tensor::{ElementConversion, Tensor};

/// Configuration to create momentum [Momentum](Momentum).
#[derive(Config)]
pub struct MomentumConfig {
    /// Momemtum factor
    #[config(default = 0.9)]
    pub momentum: f64,
    /// Dampening factor.
    #[config(default = 0.1)]
    pub dampening: f64,
    /// Enables Nesterov momentum, see [On the importance of initialization and momentum in deep learning](http://www.cs.toronto.edu/~hinton/absps/momentum.pdf).
    #[config(default = false)]
    pub nesterov: bool,
}

/// Momemtum implementation that transforms gradients.
pub struct Momentum<B: ADBackend> {
    momentum: B::Elem,
    dampening: f64,
    nesterov: bool,
    velocity: GradientsParams<B>,
}

impl<B: ADBackend> Momentum<B> {
    pub fn new(config: &MomentumConfig) -> Self {
        Self {
            momentum: config.momentum.to_elem(),
            dampening: config.dampening,
            velocity: GradientsParams::<B>::new(),
            nesterov: config.nesterov,
        }
    }

    pub fn transform<const D: usize>(
        &mut self,
        id: &ParamId,
        grad: Tensor<B::InnerBackend, D>,
    ) -> Tensor<B::InnerBackend, D> {
        let velocity = match self.velocity.get::<D>(id) {
            Some(grad_last_step) => grad
                .clone()
                .mul_scalar(1.0 - self.dampening)
                .add(grad_last_step.mul_scalar(self.momentum)),
            None => grad.clone(),
        };

        let output = match self.nesterov {
            true => velocity.clone().mul_scalar(self.momentum).add(grad),
            false => velocity.clone(),
        };

        // Update velocity
        self.velocity.register(id.clone(), velocity);

        output
    }

    pub fn register_state<const D: usize>(&self, id: &ParamId, state: &mut StateNamed<B::Elem>) {
        register_state_gradients::<D, B, _>(id, state, &self.velocity, Self::state_key);
    }

    pub fn load_state<const D: usize>(
        &mut self,
        id: &ParamId,
        state: &StateNamed<B::Elem>,
        device: &B::Device,
    ) {
        load_state_gradients::<D, B, _>(id, state, &mut self.velocity, Self::state_key, device);
    }

    fn state_key(id: &ParamId) -> String {
        format!("momentum-{id}")
    }
}
