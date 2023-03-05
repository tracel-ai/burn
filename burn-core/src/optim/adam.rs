use crate as burn;

use super::{
    decay::{WeightDecay, WeightDecayConfig},
    load_state_gradients, register_state_gradients, GradientsParams,
};
use crate::config::Config;
use crate::module::{ParamId, StateNamed};
use crate::optim::Optimizer;
use crate::tensor::{backend::ADBackend, Tensor};
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
    #[config(default = 1e-5)]
    epsilon: f32,
    /// [Weight decay](WeightDecayConfig) config.
    pub weight_decay: Option<WeightDecayConfig>,
}

/// Adam optimizer as described in the paper [Adam: A Method for Stochastic Optimization](https://arxiv.org/pdf/1412.6980.pdf).
pub struct Adam<B: ADBackend> {
    learning_rate: B::FloatElem,
    momentum: AdaptiveMomentum,
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
                time: GradientsParams::new(),
                moment_1: GradientsParams::new(),
                moment_2: GradientsParams::new(),
            },
            weight_decay: config
                .weight_decay
                .as_ref()
                .map(|config| WeightDecay::new(config)),
        }
    }
}

impl<B: ADBackend> Optimizer for Adam<B> {
    type Backend = B;

    fn update_tensor<const D: usize>(
        &mut self,
        id: &ParamId,
        tensor: Tensor<B, D>,
        grad: Tensor<B::InnerBackend, D>,
    ) -> Tensor<B, D> {
        let grad = match &mut self.weight_decay {
            Some(weight_decay) => weight_decay.transform(id, grad),
            None => grad,
        };
        let grad = self.momentum.transform::<B, D>(id, grad);
        let delta = grad.mul_scalar(self.learning_rate);

        Tensor::from_inner(tensor.inner() - delta)
    }

    fn register_param_state<const D: usize>(
        &self,
        id: &ParamId,
        state: &mut StateNamed<B::FloatElem>,
    ) {
        self.momentum.register_state::<B, D>(id, state);

        if let Some(weight_decay) = &self.weight_decay {
            weight_decay.register_state::<D>(id, state);
        }
    }

    fn load_param_state<const D: usize>(
        &mut self,
        id: &ParamId,
        state: &StateNamed<B::FloatElem>,
        device: &B::Device,
    ) {
        self.momentum.load_state::<B, D>(id, state, device);

        if let Some(weight_decay) = &mut self.weight_decay {
            weight_decay.load_state::<D>(id, state, device);
        }
    }
}

struct AdaptiveMomentum {
    beta_1: f32,
    beta_2: f32,
    epsilon: f32,
    time: GradientsParams,
    moment_1: GradientsParams,
    moment_2: GradientsParams,
}

impl AdaptiveMomentum {
    pub fn transform<B: ADBackend, const D: usize>(
        &mut self,
        id: &ParamId,
        grad: Tensor<B::InnerBackend, D>,
    ) -> Tensor<B::InnerBackend, D> {
        let factor = 1.0 - self.beta_1;
        let moment_1 = match self.moment_1.remove::<B::InnerBackend, D>(id) {
            Some(moment_last_step) => moment_last_step
                .mul_scalar(self.beta_1)
                .add(grad.clone().mul_scalar(factor)),
            None => grad.clone().mul_scalar(factor),
        };

        let factor = 1.0 - self.beta_2;
        let moment_2 = match self.moment_2.remove::<B::InnerBackend, D>(id) {
            Some(moment_last_step) => moment_last_step
                .mul_scalar(self.beta_2)
                .add(grad.powf(2.0).mul_scalar(factor)),
            None => grad.powf(2.0).mul_scalar(factor),
        };

        let time = match self.time.remove::<B::InnerBackend, 1>(id) {
            Some(time) => time.add_scalar(1),
            None => Tensor::ones([1]),
        };

        self.moment_1.register(id.clone(), moment_1.clone());
        self.moment_2.register(id.clone(), moment_2.clone());
        self.time.register(id.clone(), time.clone());

        let time = time.single_value().to_elem();
        let moment_1_corrected = moment_1.div_scalar(1f32 - self.beta_1.powf(time));
        let moment_2_corrected = moment_2.div_scalar(1f32 - self.beta_2.powf(time));

        moment_1_corrected.div(moment_2_corrected.sqrt().add_scalar(self.epsilon))
    }

    pub fn register_state<B: ADBackend, const D: usize>(
        &self,
        id: &ParamId,
        state: &mut StateNamed<B::FloatElem>,
    ) {
        register_state_gradients::<D, B, _>(id, state, &self.moment_1, Self::state_key_1);
        register_state_gradients::<D, B, _>(id, state, &self.moment_2, Self::state_key_2);
        register_state_gradients::<1, B, _>(id, state, &self.time, Self::state_key_time);
    }

    pub fn load_state<B: ADBackend, const D: usize>(
        &mut self,
        id: &ParamId,
        state: &StateNamed<B::FloatElem>,
        device: &B::Device,
    ) {
        load_state_gradients::<D, B, _>(id, state, &mut self.moment_1, Self::state_key_1, device);
        load_state_gradients::<D, B, _>(id, state, &mut self.moment_2, Self::state_key_2, device);
        load_state_gradients::<1, B, _>(id, state, &mut self.time, Self::state_key_time, device);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::module::{Module, State};
    use crate::tensor::{Data, Distribution, Tensor};
    use crate::{nn, TestADBackend};

    #[test]
    fn test_adam_optimizer_save_load_state() {
        let linear = nn::Linear::new(&nn::LinearConfig::new(6, 6));
        let x = Tensor::<TestADBackend, 2>::random([2, 6], Distribution::Standard);
        let mut optimizer = Adam::new(&AdamConfig::new(0.01));
        let grads = linear.forward(x).backward();
        let grads = GradientsParams::from_grads(grads, &linear);
        let linear = optimizer.update_module(linear, grads);

        let state_optim_before = optimizer.state(&linear);
        let mut optimizer = Adam::new(&AdamConfig::new(0.01));
        optimizer.load(&linear, &state_optim_before).unwrap();
        let state_optim_after = optimizer.state(&linear);

        assert_eq!(state_optim_before, state_optim_after);
    }

    #[test]
    fn test_adam_optimizer_with_numbers() {
        let linear = given_linear_layer(
            Data::from([
                [-0.3206, 0.1374, 0.4043, 0.3200, 0.0859, 0.0671],
                [0.0777, -0.0185, -0.3667, 0.2550, 0.1955, -0.2922],
                [-0.0190, 0.0346, -0.2962, 0.2484, -0.2780, 0.3130],
                [-0.2980, -0.2214, -0.3715, -0.2981, -0.0761, 0.1626],
                [0.3300, -0.2182, 0.3717, -0.1729, 0.3796, -0.0304],
                [-0.0159, -0.0120, 0.1258, 0.1921, 0.0293, 0.3833],
            ]),
            Data::from([-0.3905, 0.0884, -0.0970, 0.1176, 0.1366, 0.0130]),
        );
        let x_1 = Tensor::from_floats([
            [0.6294, 0.0940, 0.8176, 0.8824, 0.5228, 0.4310],
            [0.7152, 0.9559, 0.7893, 0.5684, 0.5939, 0.8883],
        ])
        .require_grad();
        let x_2 = Tensor::from_floats([
            [0.8491, 0.2108, 0.8939, 0.4433, 0.5527, 0.2528],
            [0.3270, 0.0412, 0.5538, 0.9605, 0.3195, 0.9085],
        ])
        .require_grad();
        let mut optimizer = Adam::new(
            &AdamConfig::new(0.01)
                .with_epsilon(1e-8)
                .with_beta_1(0.9)
                .with_beta_2(0.999),
        );

        let grads = linear.forward(x_1).backward();
        let grads = GradientsParams::from_grads(grads, &linear);
        let linear = optimizer.update_module(linear, grads);

        let grads = linear.forward(x_2).backward();
        let grads = GradientsParams::from_grads(grads, &linear);
        let linear = optimizer.update_module(linear, grads);

        let state_updated = linear.state();
        let state_expected = given_linear_state(
            Data::from([
                [-0.3405, 0.1191, 0.3843, 0.3000, 0.0661, 0.0471],
                [0.0577, -0.0367, -0.3846, 0.2360, 0.1756, -0.3122],
                [-0.0389, 0.0150, -0.3161, 0.2284, -0.2978, 0.2930],
                [-0.3180, -0.2396, -0.3915, -0.3181, -0.0960, 0.1427],
                [0.3100, -0.2365, 0.3517, -0.1929, 0.3597, -0.0504],
                [-0.0358, -0.0303, 0.1059, 0.1721, 0.0095, 0.3634],
            ]),
            Data::from([-0.4105, 0.0684, -0.1170, 0.0976, 0.1166, -0.0070]),
        );
        let (weight_updated, bias_updated) = extract_tensor(state_updated);
        let (weight_expected, bias_expected) = extract_tensor(state_expected);

        bias_updated.assert_approx_eq(&bias_expected, 2);
        weight_updated.assert_approx_eq(&weight_expected, 2);
    }

    fn given_linear_layer(weight: Data<f32, 2>, bias: Data<f32, 1>) -> nn::Linear<TestADBackend> {
        let linear = nn::Linear::new(&nn::LinearConfig::new(6, 6));
        let state = given_linear_state(weight, bias);

        linear.load(&state).unwrap()
    }

    fn given_linear_state(weight: Data<f32, 2>, bias: Data<f32, 1>) -> State<f32> {
        let mut state = StateNamed::<f32>::new();
        let mut state_weight = StateNamed::<f32>::new();
        state_weight.register_state("data", State::Data(weight.serialize()));
        state_weight.register_state("id", State::ParamId(ParamId::from("weight_param_id")));

        let mut state_bias = StateNamed::<f32>::new();
        state_bias.register_state("data", State::Data(bias.serialize()));
        state_bias.register_state("id", State::ParamId(ParamId::from("bias_param_id")));

        state.register_state("weight", State::StateNamed(state_weight));
        state.register_state("bias", State::StateNamed(state_bias));

        State::StateNamed(state)
    }

    fn extract_tensor(state: State<f32>) -> (Data<f32, 2>, Data<f32, 1>) {
        let mut state = match state {
            State::StateNamed(val) => val.values,
            _ => panic!("Should be state name with key 'weight' and 'bias'"),
        };
        let state_weight = state.remove("weight").expect("Contains weight key");
        let state_bias = state.remove("bias").expect("Contains weight key");

        let weights = match state_weight {
            State::StateNamed(mut val) => val.values.remove("data").expect("Should contains data"),
            _ => panic!("Should be state name with key 'value' and 'id'"),
        };
        let bias = match state_bias {
            State::StateNamed(mut val) => val.values.remove("data").expect("Should contains data"),
            _ => panic!("Should be state name with key 'value' and 'id'"),
        };

        let weights = match weights {
            State::Data(data) => Data::from(data),
            _ => panic!("Should be data"),
        };
        let bias = match bias {
            State::Data(data) => Data::from(data),
            _ => panic!("Should be data"),
        };

        (weights, bias)
    }
}
