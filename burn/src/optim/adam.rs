use crate as burn;

use super::{
    decay::{WeightDecay, WeightDecayConfig},
    load_state_gradients, register_state_gradients,
    visitor::GradientsParams,
};
use crate::config::Config;
use crate::module::{ParamId, StateNamed};
use crate::optim::Optimizer;
use burn::tensor::{backend::ADBackend, Tensor};
use burn_tensor::ElementConversion;

#[derive(Config)]
pub struct AdamConfig {
    /// Learning rate for the optimizer.
    learning_rate: f64,
    /// Parameter for Adam.
    #[config(default = 0.9)]
    beta_1: f32,
    /// Parameter for Adam.
    #[config(default = 0.999)]
    beta_2: f32,
    /// A value required for numerical stability.
    #[config(default = 1e-8)]
    epsilon: f32,
    /// [Weight decay](WeightDecayConfig) config.
    pub weight_decay: Option<WeightDecayConfig>,
}

/// Adam optimizer as described in the paper [Adam: A Method for Stochastic Optimization](https://arxiv.org/pdf/1412.6980.pdf).
pub struct Adam<B: ADBackend> {
    learning_rate: B::Elem,
    momentum: AdaptiveMomentum<B>,
    weight_decay: Option<WeightDecay<B>>,
}

impl<B: ADBackend> Adam<B> {
    pub fn new(config: &AdamConfig) -> Self {
        Self {
            learning_rate: config.learning_rate.to_elem(),
            momentum: AdaptiveMomentum {
                beta_1: config.beta_1,
                beta_2: config.beta_2,
                epsilon: config.epsilon,
                time: GradientsParams::<B>::new(),
                moment_1: GradientsParams::<B>::new(),
                moment_2: GradientsParams::<B>::new(),
            },
            weight_decay: match &config.weight_decay {
                Some(config) => Some(WeightDecay::new(config)),
                None => None,
            },
        }
    }
}

impl<B: ADBackend> Optimizer for Adam<B> {
    type Backend = B;

    fn update_tensor<const D: usize>(
        &mut self,
        id: &ParamId,
        tensor: &mut Tensor<B, D>,
        grad: Tensor<B::InnerBackend, D>,
    ) {
        let grad = match &mut self.weight_decay {
            Some(weight_decay) => weight_decay.transform(id, grad),
            None => grad,
        };
        let grad = self.momentum.transform(id, grad);

        let delta = grad.mul_scalar(self.learning_rate);
        tensor.update(tensor.inner() - delta);
    }

    fn register_param_state<const D: usize>(&self, id: &ParamId, state: &mut StateNamed<B::Elem>) {
        self.momentum.register_state::<D>(id, state);

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
        self.momentum.load_state::<D>(id, state, device);

        if let Some(weight_decay) = &mut self.weight_decay {
            weight_decay.load_state::<D>(id, state, device);
        }
    }
}

struct AdaptiveMomentum<B: ADBackend> {
    beta_1: f32,
    beta_2: f32,
    epsilon: f32,
    time: GradientsParams<B>,
    moment_1: GradientsParams<B>,
    moment_2: GradientsParams<B>,
}

impl<B: ADBackend> AdaptiveMomentum<B> {
    pub fn transform<const D: usize>(
        &mut self,
        id: &ParamId,
        grad: Tensor<B::InnerBackend, D>,
    ) -> Tensor<B::InnerBackend, D> {
        let factor = 1.0 - self.beta_1;
        let moment_1 = match self.moment_1.get::<D>(id) {
            Some(moment_last_step) => moment_last_step
                .mul_scalar(self.beta_1)
                .add(&grad.mul_scalar(factor)),
            None => grad.mul_scalar(factor),
        };

        let factor = 1.0 - self.beta_2;
        let moment_2 = match self.moment_2.get::<D>(id) {
            Some(moment_last_step) => moment_last_step
                .mul_scalar(self.beta_2)
                .add(&grad.powf(2.0).mul_scalar(factor)),
            None => grad.powf(2.0).mul_scalar(factor),
        };

        let time = match self.time.get::<1>(id) {
            Some(time) => time.add_scalar(1),
            None => Tensor::ones([1]),
        };

        self.moment_1.register(id.clone(), moment_1.clone());
        self.moment_2.register(id.clone(), moment_2.clone());
        self.time.register(id.clone(), time.clone());

        let time = time.single_value().to_elem();
        let moment_1_corrected = moment_1.div_scalar(1f32 - self.beta_1.powf(time));
        let moment_2_corrected = moment_2.div_scalar(1f32 - self.beta_2.powf(time));

        moment_1_corrected.div(&moment_2_corrected.sqrt().add_scalar(self.epsilon))
    }

    pub fn register_state<const D: usize>(&self, id: &ParamId, state: &mut StateNamed<B::Elem>) {
        register_state_gradients::<D, B, _>(id, state, &self.moment_1, Self::state_key_1);
        register_state_gradients::<D, B, _>(id, state, &self.moment_2, Self::state_key_2);
        register_state_gradients::<D, B, _>(id, state, &self.time, Self::state_key_time);
    }

    pub fn load_state<const D: usize>(
        &mut self,
        id: &ParamId,
        state: &StateNamed<B::Elem>,
        device: &B::Device,
    ) {
        load_state_gradients::<D, B, _>(id, state, &mut self.moment_1, Self::state_key_1, device);
        load_state_gradients::<D, B, _>(id, state, &mut self.moment_2, Self::state_key_2, device);
        load_state_gradients::<D, B, _>(id, state, &mut self.time, Self::state_key_time, device);
    }

    fn state_key_1(id: &ParamId) -> String {
        format!("moment_1-{id}")
    }

    fn state_key_2(id: &ParamId) -> String {
        format!("moment_2-{id}")
    }

    fn state_key_time(id: &ParamId) -> String {
        format!("time-{id}")
    }
}
