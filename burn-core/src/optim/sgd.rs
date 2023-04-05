use crate as burn;
use crate::module::ADModule;

use super::decay::{WeightDecay, WeightDecayConfig, WeightDecayState};
use super::momentum::{MomemtumState, Momentum, MomentumConfig};
use super::SimpleOptimizer;
use crate::config::Config;
use crate::optim::adaptor::OptimizerAdaptor;
use crate::record::Record;
use crate::tensor::{ElementConversion, Tensor};
use burn_tensor::backend::{ADBackend, Backend};

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

impl SgdConfig {
    pub fn init<B: ADBackend, M: ADModule<B>>(
        &self,
    ) -> OptimizerAdaptor<Sgd<B::InnerBackend>, M, B> {
        let learning_rate = self.learning_rate.elem();
        let momentum = self.momentum.as_ref().map(Momentum::new);
        let weight_decay = self.weight_decay.as_ref().map(WeightDecay::new);

        Sgd {
            learning_rate,
            momentum,
            weight_decay,
        }
        .into()
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

    fn to_device<const D: usize>(mut state: Self::State<D>, device: &B::Device) -> Self::State<D> {
        state.weight_decay = state.weight_decay.map(|state| state.to_device(device));
        state.momentum = state.momentum.map(|state| state.to_device(device));
        state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        nn::{Linear, LinearConfig},
        optim::{GradientsParams, Optimizer},
        tensor::{Distribution, Shape},
        TestADBackend, TestBackend,
    };

    #[test]
    fn with_updated_params_should_have_state() {
        let layer = layer();
        let mut optim = sgd_with_all();
        let loss = layer.forward(random_tensor());
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &layer);
        let _layer = optim.step(layer, grads);

        let record = optim.to_record();

        assert!(!record.is_empty());
    }

    #[test]
    fn without_updated_params_should_not_have_state() {
        let optim = sgd_with_all();
        let record = optim.to_record();
        assert!(record.is_empty());
    }

    #[test]
    fn should_load_state() {
        let layer = layer();
        let mut optim = sgd_with_all();
        let loss = layer.forward(random_tensor());
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &layer);
        let _layer = optim.step(layer, grads);

        let record = optim.to_record();
        let optim_new = sgd_with_all();
        let record_new = optim_new.to_record();
        let optim_new = optim_new.load_record(record.clone());
        let state_restored = optim_new.to_record();

        assert_ne!(record.len(), record_new.len());
        assert_eq!(record.len(), state_restored.len());
    }

    fn random_tensor() -> Tensor<TestADBackend, 2> {
        Tensor::<TestADBackend, 2>::random(Shape::new([2, 20]), Distribution::Standard)
    }

    fn layer() -> Linear<TestADBackend> {
        LinearConfig::new(20, 20).with_bias(true).init()
    }

    fn sgd_with_all() -> OptimizerAdaptor<Sgd<TestBackend>, Linear<TestADBackend>, TestADBackend> {
        SgdConfig {
            learning_rate: 0.02,
            weight_decay: Some(WeightDecayConfig { penalty: 0.05 }),
            momentum: Some(MomentumConfig {
                momentum: 0.9,
                dampening: 0.1,
                nesterov: true,
            }),
        }
        .init()
    }
}
