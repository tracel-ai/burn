use crate::macros::config;
use crate::module::ParamId;
use crate::tensor::backend::ADBackend;
use crate::tensor::{ElementConversion, Gradients, Tensor};

config!(
    /// Configuration to create [WeightDecay](WeightDecay).
    pub struct WeightDecayConfig {
        /// L2 penalty.
        pub penalty: f64,
    }
);

/// Weight decay implementation that transforms gradients.
pub struct WeightDecay<B: ADBackend> {
    penalty: B::Elem,
    gradients: Gradients,
}

impl<B: ADBackend> WeightDecay<B> {
    pub fn new(config: &WeightDecayConfig) -> Self {
        Self {
            penalty: config.penalty.to_elem(),
            gradients: Gradients::empty(),
        }
    }

    pub fn transform<const D: usize>(
        &mut self,
        id: &ParamId,
        grad: Tensor<B::InnerBackend, D>,
    ) -> Tensor<B::InnerBackend, D> {
        let id = id.to_string();

        match self.gradients.get::<Tensor<B::InnerBackend, D>>(&id) {
            Some(grad_last_step) => grad_last_step.mul_scalar(&self.penalty).add(&grad),
            None => grad,
        }
    }
}
