use super::momentum::{Momentum, MomentumConfig};
use crate::macros::config;
use crate::module::ParamId;
use crate::optim::Optimizer;
use crate::tensor::backend::ADBackend;
use crate::tensor::{ElementConversion, Gradients, Tensor};

config!(
    /// Configuration to create the [Sgd](Sgd) optimizer.
    pub struct SgdConfig {
        /// Learning rate for the optimizer.
        pub learning_rate: f64,
        /// Weight decay, L2 penalty.
        pub weight_decay: f64,
        /// [Momentum](MomentumConfig) config.
        pub momentum: Option<MomentumConfig>,
    }
);

/// Optimizer that implements stochastic gradient descent with momentum.
///
/// Momentum is optinal and can be [configured](SgdConfig::momentum).
pub struct Sgd<B: ADBackend> {
    learning_rate: B::Elem,
    momentum: Option<Momentum<B>>,
}

impl<B: ADBackend> Sgd<B> {
    pub fn new(config: &SgdConfig) -> Self {
        let learning_rate = config.learning_rate.to_elem();
        let momentum = config.momentum.as_ref().map(|config| Momentum::new(config));

        Self {
            learning_rate,
            momentum,
        }
    }
}
impl<B: ADBackend> Optimizer for Sgd<B> {
    type Backend = B;

    fn update<const D: usize>(
        &mut self,
        id: &ParamId,
        tensor: &mut Tensor<B, D>,
        grads: &Gradients,
    ) {
        let grad = tensor.grad(grads).unwrap();
        let grad = match &mut self.momentum {
            Some(momentum) => momentum.transform(id, grad),
            None => grad,
        };

        let delta = grad.mul_scalar(&self.learning_rate);
        tensor.update(tensor.inner() - delta);
    }
}
