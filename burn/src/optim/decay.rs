use crate as burn;

use super::visitor::GradientsParams;
use super::{load_state_gradients, register_state_gradients};
use crate::config::Config;
use crate::module::{ParamId, StateNamed};
use crate::tensor::backend::ADBackend;
use crate::tensor::{ElementConversion, Tensor};

/// Configuration to create [WeightDecay](WeightDecay).
#[derive(Config)]
pub struct WeightDecayConfig {
    /// L2 penalty.
    pub penalty: f64,
}

/// Weight decay implementation that transforms gradients.
pub struct WeightDecay<B: ADBackend> {
    penalty: B::Elem,
    gradients: GradientsParams<B>,
}

impl<B: ADBackend> WeightDecay<B> {
    pub fn new(config: &WeightDecayConfig) -> Self {
        Self {
            penalty: config.penalty.to_elem(),
            gradients: GradientsParams::<B>::new(),
        }
    }

    pub fn transform<const D: usize>(
        &mut self,
        id: &ParamId,
        grad: Tensor<B::InnerBackend, D>,
    ) -> Tensor<B::InnerBackend, D> {
        let grad = match self.gradients.get::<D>(id) {
            Some(grad_last_step) => grad_last_step.mul_scalar(self.penalty).add(&grad),
            None => grad,
        };

        // Update gradients
        self.gradients.register(id.clone(), grad.clone());

        grad
    }
    pub fn register_state<const D: usize>(&self, id: &ParamId, state: &mut StateNamed<B::Elem>) {
        register_state_gradients::<D, B, _>(id, state, &self.gradients, Self::state_key);
    }

    pub fn load_state<const D: usize>(
        &mut self,
        id: &ParamId,
        state: &StateNamed<B::Elem>,
        device: &B::Device,
    ) {
        load_state_gradients::<D, B, _>(id, state, &mut self.gradients, Self::state_key, device);
    }

    fn state_key(id: &ParamId) -> String {
        format!("weight-decay-{id}")
    }
}
