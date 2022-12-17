use crate as burn;

use super::decay::{WeightDecay, WeightDecayConfig};
use super::momentum::{Momentum, MomentumConfig};
use crate::config::Config;
use crate::module::{ParamId, StateNamed};
use crate::optim::Optimizer;
use crate::tensor::backend::ADBackend;
use crate::tensor::{ElementConversion, Tensor};

/// Configuration to create the [Sgd](Sgd) optimizer.
#[derive(Config)]
pub struct SgdConfig {
    /// Learning rate for the optimizer.
    #[config(default = 0.01)]
    pub learning_rate: f64,
    /// [Weight decay](WeightDecayConfig) config.
    pub weight_decay: Option<WeightDecayConfig>,
    /// [Momentum](MomentumConfig) config.
    pub momentum: Option<MomentumConfig>,
}

/// Optimizer that implements stochastic gradient descent with momentum.
///
/// Momentum is optinal and can be [configured](SgdConfig::momentum).
pub struct Sgd<B: ADBackend> {
    learning_rate: B::Elem,
    momentum: Option<Momentum<B>>,
    weight_decay: Option<WeightDecay<B>>,
}

impl<B: ADBackend> Sgd<B> {
    pub fn new(config: &SgdConfig) -> Self {
        let learning_rate = config.learning_rate.to_elem();
        let momentum = config.momentum.as_ref().map(|config| Momentum::new(config));
        let weight_decay = config
            .weight_decay
            .as_ref()
            .map(|config| WeightDecay::new(config));

        Self {
            learning_rate,
            momentum,
            weight_decay,
        }
    }
}

impl<B: ADBackend> Optimizer for Sgd<B> {
    type Backend = B;

    fn update_tensor<const D: usize>(
        &mut self,
        id: &ParamId,
        tensor: &mut Tensor<B, D>,
        grads: &B::Gradients,
    ) {
        if let Some(grad) = tensor.grad(grads) {
            let grad = match &mut self.weight_decay {
                Some(weight_decay) => weight_decay.transform(id, grad),
                None => grad,
            };
            let grad = match &mut self.momentum {
                Some(momentum) => momentum.transform(id, grad),
                None => grad,
            };

            let delta = grad.mul_scalar(self.learning_rate);
            tensor.update(tensor.inner() - delta);
        }
    }

    fn register_param_state<const D: usize>(&self, id: &ParamId, state: &mut StateNamed<B::Elem>) {
        if let Some(momentum) = &self.momentum {
            momentum.register_state::<D>(id, state);
        }

        if let Some(weight_decay) = &self.weight_decay {
            weight_decay.register_state::<D>(id, state);
        }
    }

    fn load_param_state<const D: usize>(
        &mut self,
        id: &ParamId,
        state: &StateNamed<B::Elem>,
        device: &B::Device,
    ) {
        if let Some(momentum) = &mut self.momentum {
            momentum.load_state::<D>(id, state, device);
        }

        if let Some(weight_decay) = &mut self.weight_decay {
            weight_decay.load_state::<D>(id, state, device);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        nn::{Linear, LinearConfig},
        tensor::{Distribution, Shape},
        TestADBackend,
    };

    #[test]
    fn with_updated_params_should_have_state() {
        let mut layer = layer();
        let mut optim = sgd_with_all();
        let loss = layer.forward(random_tensor());
        let grads = loss.backward();
        optim.update_module(&mut layer, &grads);

        let state = optim.state(&layer);

        assert!(!state.is_empty());
    }

    #[test]
    fn without_updated_params_should_not_have_state() {
        let layer = layer();
        let optim = sgd_with_all();

        let state = optim.state(&layer);

        assert!(state.is_empty());
    }

    #[test]
    fn without_momentum_and_weights_decay_should_not_have_state() {
        let mut layer = layer();
        let mut optim = sgd_with_nothing();
        let loss = layer.forward(random_tensor());
        let grads = loss.backward();
        optim.update_module(&mut layer, &grads);

        let state = optim.state(&layer);

        assert!(state.is_empty());
    }

    #[test]
    fn should_load_state() {
        let mut layer = layer();
        let mut optim = sgd_with_all();
        let loss = layer.forward(random_tensor());
        let grads = loss.backward();
        optim.update_module(&mut layer, &grads);

        let state = optim.state(&layer);
        let mut optim_new = sgd_with_all();
        let state_new = optim_new.state(&layer);
        optim_new.load(&layer, &state).unwrap();
        let state_restored = optim_new.state(&layer);

        assert_ne!(state, state_new);
        assert_eq!(state, state_restored);
    }

    fn random_tensor() -> Tensor<TestADBackend, 2> {
        Tensor::<TestADBackend, 2>::random(Shape::new([2, 20]), Distribution::Standard)
    }

    fn layer() -> Linear<TestADBackend> {
        Linear::<TestADBackend>::new(&LinearConfig {
            d_input: 20,
            d_output: 20,
            bias: true,
        })
    }

    fn sgd_with_all() -> Sgd<TestADBackend> {
        Sgd::new(&SgdConfig {
            learning_rate: 0.02,
            weight_decay: Some(WeightDecayConfig { penalty: 0.05 }),
            momentum: Some(MomentumConfig {
                momentum: 0.9,
                dampening: 0.1,
                nesterov: true,
            }),
        })
    }

    fn sgd_with_nothing() -> Sgd<TestADBackend> {
        Sgd::new(&SgdConfig {
            learning_rate: 0.02,
            weight_decay: None,
            momentum: None,
        })
    }
}
