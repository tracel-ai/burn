use crate as burn;

use super::decay::{WeightDecay, WeightDecayConfig, WeightDecayState};
use super::momentum::{MomemtumState, Momentum, MomentumConfig};
use super::SimpleOptimizer;
use crate::config::Config;
use crate::record::Record;
use crate::tensor::{ElementConversion, Tensor};
use burn_tensor::backend::Backend;

/// Configuration to create the [Sgd](Sgd) optimizer.
#[derive(Config)]
pub struct SgdConfig {
    /// Learning rate for the optimizer.
    pub learning_rate: f64,
    /// [Weight decay](WeightDecayConfig) config.
    pub weight_decay: Option<WeightDecayConfig>,
    /// [Momentum](MomentumConfig) config.
    pub momentum: Option<MomentumConfig>,
}

/// Optimizer that implements stochastic gradient descent with momentum.
///
/// Momentum is optinal and can be [configured](SgdConfig::momentum).
pub struct Sgd<B: Backend> {
    learning_rate: B::FloatElem,
    momentum: Option<Momentum<B>>,
    weight_decay: Option<WeightDecay<B>>,
}

#[derive(Record, Clone, new)]
pub struct SgdState<B: Backend, const D: usize> {
    weight_decay: Option<WeightDecayState<B, D>>,
    momentum: Option<MomemtumState<B, D>>,
}

impl<B: Backend> Sgd<B> {
    pub fn new(config: &SgdConfig) -> Self {
        let learning_rate = config.learning_rate.elem();
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

impl<B: Backend> SimpleOptimizer<B> for Sgd<B> {
    type State<const D: usize> = SgdState<B, D>;

    fn step<const D: usize>(
        &self,
        tensor: Tensor<B, D>,
        mut grad: Tensor<B, D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<B, D>, Option<Self::State<D>>) {
        let mut state_weight_decay = None;
        let mut state_momemtum = None;

        if let Some(state) = state {
            state_weight_decay = state.weight_decay;
            state_momemtum = state.momentum;
        }

        if let Some(weight_decay) = &self.weight_decay {
            let (grad_out, state) = weight_decay.transform(grad, state_weight_decay);
            state_weight_decay = Some(state);
            grad = grad_out;
        }

        if let Some(momentum) = &self.momentum {
            let (grad_out, state) = momentum.transform(grad, state_momemtum);
            state_momemtum = Some(state);
            grad = grad_out;
        }

        let state = SgdState::new(state_weight_decay, state_momemtum);
        let delta = grad.mul_scalar(self.learning_rate);

        (tensor - delta, Some(state))
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::{
//         nn::{Linear, LinearConfig},
//         optim::GradientsParams,
//         tensor::{Distribution, Shape},
//         TestADBackend,
//     };
//
//     #[test]
//     fn with_updated_params_should_have_state() {
//         let layer = layer();
//         let mut optim = sgd_with_all();
//         let loss = layer.forward(random_tensor());
//         let grads = loss.backward();
//         let grads = GradientsParams::from_grads(grads, &layer);
//         let layer = optim.update_module(layer, grads);
//
//         let state = optim.state(&layer);
//
//         assert!(!state.is_empty());
//     }
//
//     #[test]
//     fn without_updated_params_should_not_have_state() {
//         let layer = layer();
//         let optim = sgd_with_all();
//
//         let state = optim.state(&layer);
//
//         assert!(state.is_empty());
//     }
//
//     #[test]
//     fn without_momentum_and_weights_decay_should_not_have_state() {
//         let layer = layer();
//         let mut optim = sgd_with_nothing();
//         let loss = layer.forward(random_tensor());
//         let grads = loss.backward();
//         let grads = GradientsParams::from_grads(grads, &layer);
//
//         let layer = optim.update_module(layer, grads);
//
//         let state = optim.state(&layer);
//
//         assert!(state.is_empty());
//     }
//
//     #[test]
//     fn should_load_state() {
//         let layer = layer();
//         let mut optim = sgd_with_all();
//         let loss = layer.forward(random_tensor());
//         let grads = loss.backward();
//         let grads = GradientsParams::from_grads(grads, &layer);
//         let layer = optim.update_module(layer, grads);
//
//         let state = optim.state(&layer);
//         let mut optim_new = sgd_with_all();
//         let state_new = optim_new.state(&layer);
//         optim_new.load(&layer, &state).unwrap();
//         let state_restored = optim_new.state(&layer);
//
//         assert_ne!(state, state_new);
//         assert_eq!(state, state_restored);
//     }
//
//     fn random_tensor() -> Tensor<TestADBackend, 2> {
//         Tensor::<TestADBackend, 2>::random(Shape::new([2, 20]), Distribution::Standard)
//     }
//
//     fn layer() -> Linear<TestADBackend> {
//         LinearConfig::new(20, 20).with_bias(true).init()
//     }
//
//     fn sgd_with_all() -> Sgd<TestADBackend> {
//         Sgd::new(&SgdConfig {
//             learning_rate: 0.02,
//             weight_decay: Some(WeightDecayConfig { penalty: 0.05 }),
//             momentum: Some(MomentumConfig {
//                 momentum: 0.9,
//                 dampening: 0.1,
//                 nesterov: true,
//             }),
//         })
//     }
//
//     fn sgd_with_nothing() -> Sgd<TestADBackend> {
//         Sgd::new(&SgdConfig {
//             learning_rate: 0.02,
//             weight_decay: None,
//             momentum: None,
//         })
//     }
// }
