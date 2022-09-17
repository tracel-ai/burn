use crate::macros::config;
use crate::module::ParamId;
use crate::tensor::backend::ADBackend;
use crate::tensor::{ElementConversion, Gradients, Tensor};

config!(
    /// Configuration to create momentum [Momentum](Momentum).
    pub struct MomentumConfig {
        /// Momemtum factor
        pub momentum: f64,
        /// Dampening factor.
        pub dampening: f64,
        /// Enables Nesterov momentum, see [On the importance of initialization and momentum in deep learning](http://www.cs.toronto.edu/~hinton/absps/momentum.pdf).
        pub nesterov: bool,
    }
);

/// Momemtum implementation that transforms gradients.
pub struct Momentum<B: ADBackend> {
    momentum: B::Elem,
    dampening: f64,
    nesterov: bool,
    velocity: Gradients,
}

impl<B: ADBackend> Momentum<B> {
    pub fn new(config: &MomentumConfig) -> Self {
        Self {
            momentum: config.momentum.to_elem(),
            dampening: config.dampening,
            velocity: Gradients::empty(),
            nesterov: config.nesterov,
        }
    }

    pub fn transform<const D: usize>(
        &mut self,
        id: &ParamId,
        grad: Tensor<B::InnerBackend, D>,
    ) -> Tensor<B::InnerBackend, D> {
        let id = id.to_string();

        let velocity = match self.velocity.get::<Tensor<B::InnerBackend, D>>(&id) {
            Some(grad_last_step) => grad
                .mul_scalar(&(1.0 - self.dampening).to_elem())
                .add(&grad_last_step.mul_scalar(&self.momentum)),
            None => grad.clone(),
        };

        // Update velocity
        self.velocity.register_any(id, velocity.clone());

        let grad = match self.nesterov {
            true => velocity.mul_scalar(&self.momentum).add(&grad),
            false => velocity,
        };

        grad
    }
}
