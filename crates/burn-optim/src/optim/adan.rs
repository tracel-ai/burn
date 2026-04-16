use burn_core as burn;

use burn::config::Config;
use burn::tensor::{Tensor, backend::AutodiffBackend};
use burn::tensor::{backend::Backend, ops::Device};
use burn::{module::AutodiffModule, record::Record};

use super::{SimpleOptimizer, adaptor::OptimizerAdaptor};
use crate::{LearningRate, grad_clipping::GradientClippingConfig};

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float as _;

/// [`Adan`] Configuration.
///
/// See:
/// - [Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models](https://arxiv.org/abs/2208.06677).
#[derive(Config, Debug)]
pub struct AdanConfig {
    /// Parameter for the first moment.
    #[config(default = 0.98)]
    beta_1: f32,
    /// Parameter for the gradient-difference momentum.
    #[config(default = 0.92)]
    beta_2: f32,
    /// Parameter for the second moment.
    #[config(default = 0.99)]
    beta_3: f32,
    /// A value required for numerical stability.
    #[config(default = 1e-8)]
    epsilon: f32,
    /// Weight decay factor.
    #[config(default = 0.0)]
    weight_decay: f32,
    /// Disable proximal weight decay and use the decoupled update instead.
    #[config(default = false)]
    no_prox: bool,
    /// [Gradient Clipping](GradientClippingConfig) config.
    grad_clipping: Option<GradientClippingConfig>,
}

/// Adan optimizer.
///
/// See:
/// - [Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models](https://arxiv.org/abs/2208.06677).
///
/// Configured by [`AdanConfig`].
#[derive(Clone)]
pub struct Adan {
    momentum: AdaptiveNesterovMomentum,
    weight_decay: f32,
    no_prox: bool,
}

/// Adan state.
#[derive(Record, Clone, new)]
pub struct AdanState<B: Backend, const D: usize> {
    /// The current adaptive Nesterov momentum state.
    pub momentum: AdaptiveNesterovMomentumState<B, D>,
}

impl<B: Backend> SimpleOptimizer<B> for Adan {
    type State<const D: usize> = AdanState<B, D>;

    fn step<const D: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<B, D>,
        grad: Tensor<B, D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<B, D>, Option<Self::State<D>>) {
        let (raw_delta, momentum_state) = self.momentum.transform(grad, state.map(|s| s.momentum));

        let decay_rate = lr * (self.weight_decay as f64);
        let delta = raw_delta.mul_scalar(lr);

        let tensor_updated = if self.no_prox {
            if decay_rate == 0.0 {
                tensor - delta
            } else {
                tensor.mul_scalar(1.0 - decay_rate) - delta
            }
        } else {
            let updated = tensor - delta;
            if decay_rate == 0.0 {
                updated
            } else {
                updated.div_scalar(1.0 + decay_rate)
            }
        };

        (tensor_updated, Some(AdanState::new(momentum_state)))
    }

    fn to_device<const D: usize>(mut state: Self::State<D>, device: &Device<B>) -> Self::State<D> {
        state.momentum = state.momentum.to_device(device);
        state
    }
}

impl AdanConfig {
    /// Build an [`Adan`] from the config.
    pub fn build(&self) -> Adan {
        Adan {
            momentum: AdaptiveNesterovMomentum {
                beta_1: self.beta_1,
                beta_2: self.beta_2,
                beta_3: self.beta_3,
                epsilon: self.epsilon,
            },
            weight_decay: self.weight_decay,
            no_prox: self.no_prox,
        }
    }

    /// Initialize Adan optimizer.
    ///
    /// # Returns
    ///
    /// Returns an optimizer that can be used to optimize a module.
    pub fn init<B: AutodiffBackend, M: AutodiffModule<B>>(&self) -> OptimizerAdaptor<Adan, M, B> {
        let mut optim = OptimizerAdaptor::from(self.build());
        if let Some(config) = &self.grad_clipping {
            optim = optim.with_grad_clipping(config.init());
        }
        optim
    }
}

/// Adaptive Nesterov momentum state.
#[derive(Record, Clone, new)]
pub struct AdaptiveNesterovMomentumState<B: Backend, const D: usize> {
    /// The number of iterations aggregated.
    pub time: usize,
    /// The first order momentum.
    pub exp_avg: Tensor<B, D>,
    /// The gradient-difference weighted second order momentum.
    pub exp_avg_sq: Tensor<B, D>,
    /// The gradient-difference momentum.
    pub exp_avg_diff: Tensor<B, D>,
    /// The negated previous gradient.
    pub neg_pre_grad: Tensor<B, D>,
}

#[derive(Clone)]
struct AdaptiveNesterovMomentum {
    beta_1: f32,
    beta_2: f32,
    beta_3: f32,
    epsilon: f32,
}

impl AdaptiveNesterovMomentum {
    pub fn transform<B: Backend, const D: usize>(
        &self,
        grad: Tensor<B, D>,
        state: Option<AdaptiveNesterovMomentumState<B, D>>,
    ) -> (Tensor<B, D>, AdaptiveNesterovMomentumState<B, D>) {
        let state = if let Some(mut state) = state {
            let grad_diff = state.neg_pre_grad.clone().add(grad.clone());
            let grad_diff_sq = grad_diff
                .clone()
                .mul_scalar(self.beta_2)
                .add(grad.clone())
                .square();

            state.exp_avg = state
                .exp_avg
                .mul_scalar(self.beta_1)
                .add(grad.clone().mul_scalar(1.0 - self.beta_1));
            state.exp_avg_diff = state
                .exp_avg_diff
                .mul_scalar(self.beta_2)
                .add(grad_diff.mul_scalar(1.0 - self.beta_2));
            state.exp_avg_sq = state
                .exp_avg_sq
                .mul_scalar(self.beta_3)
                .add(grad_diff_sq.mul_scalar(1.0 - self.beta_3));
            state.neg_pre_grad = grad.mul_scalar(-1.0);
            state.time += 1;
            state
        } else {
            AdaptiveNesterovMomentumState::new(
                1,
                grad.clone().mul_scalar(1.0 - self.beta_1),
                grad.clone().square().mul_scalar(1.0 - self.beta_3),
                grad.zeros_like(),
                grad.clone().mul_scalar(-1.0),
            )
        };

        let time = state.time as i32;
        let denom = state
            .exp_avg_sq
            .clone()
            .sqrt()
            .div_scalar((1.0 - self.beta_3.powi(time)).sqrt())
            .add_scalar(self.epsilon);
        let update = state
            .exp_avg
            .clone()
            .div_scalar(1.0 - self.beta_1.powi(time))
            .div(denom.clone())
            .add(
                state
                    .exp_avg_diff
                    .clone()
                    .mul_scalar(self.beta_2)
                    .div_scalar(1.0 - self.beta_2.powi(time))
                    .div(denom),
            );

        (update, state)
    }
}

impl<B: Backend, const D: usize> AdaptiveNesterovMomentumState<B, D> {
    #[allow(clippy::wrong_self_convention)]
    fn to_device(mut self, device: &B::Device) -> Self {
        self.exp_avg = self.exp_avg.to_device(device);
        self.exp_avg_sq = self.exp_avg_sq.to_device(device);
        self.exp_avg_diff = self.exp_avg_diff.to_device(device);
        self.neg_pre_grad = self.neg_pre_grad.to_device(device);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestAutodiffBackend;
    use crate::{GradientsParams, Optimizer};
    use burn::module::{Module, Param};
    use burn::tensor::{Distribution, Tensor, TensorData};
    use burn::tensor::{Tolerance, ops::FloatElem};
    use burn_nn::{Linear, LinearConfig, LinearRecord};

    type FT = FloatElem<TestAutodiffBackend>;

    const LEARNING_RATE: LearningRate = 0.01;

    #[test]
    fn test_adan_optimizer_save_load_state() {
        let device = Default::default();
        let linear = LinearConfig::new(6, 6).init(&device);
        let x = Tensor::<TestAutodiffBackend, 2>::random([2, 6], Distribution::Default, &device);
        let mut optimizer = create_adan();
        let grads = linear.forward(x).backward();
        let grads = GradientsParams::from_grads(grads, &linear);
        let _linear = optimizer.step(LEARNING_RATE, linear, grads);

        #[cfg(feature = "std")]
        {
            use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};

            BinFileRecorder::<FullPrecisionSettings>::default()
                .record(
                    optimizer.to_record(),
                    std::env::temp_dir().as_path().join("test_optim_adan"),
                )
                .unwrap();
        }
        #[cfg(not(feature = "std"))]
        {
            use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};

            let result = BinBytesRecorder::<FullPrecisionSettings>::default()
                .record(optimizer.to_record(), ())
                .unwrap();
            assert!(!result.is_empty());
        }

        let state_optim_before = optimizer.to_record();
        let state_optim_before_copy = optimizer.to_record();
        let optimizer = create_adan();
        let optimizer = optimizer.load_record(state_optim_before_copy);
        let state_optim_after = optimizer.to_record();

        assert_eq!(state_optim_before.len(), state_optim_after.len());
    }

    #[test]
    fn test_adan_optimizer_with_numbers() {
        let linear = given_linear_layer(
            TensorData::from([
                [-0.3206, 0.1374, 0.4043, 0.3200, 0.0859, 0.0671],
                [0.0777, -0.0185, -0.3667, 0.2550, 0.1955, -0.2922],
                [-0.0190, 0.0346, -0.2962, 0.2484, -0.2780, 0.3130],
                [-0.2980, -0.2214, -0.3715, -0.2981, -0.0761, 0.1626],
                [0.3300, -0.2182, 0.3717, -0.1729, 0.3796, -0.0304],
                [-0.0159, -0.0120, 0.1258, 0.1921, 0.0293, 0.3833],
            ]),
            TensorData::from([-0.3905, 0.0884, -0.0970, 0.1176, 0.1366, 0.0130]),
        );
        let device = Default::default();
        let x_1 = Tensor::<TestAutodiffBackend, 2>::from_floats(
            [
                [0.6294, 0.0940, 0.8176, 0.8824, 0.5228, 0.4310],
                [0.7152, 0.9559, 0.7893, 0.5684, 0.5939, 0.8883],
            ],
            &device,
        )
        .require_grad();
        let x_2 = Tensor::<TestAutodiffBackend, 2>::from_floats(
            [
                [0.8491, 0.2108, 0.8939, 0.4433, 0.5527, 0.2528],
                [0.3270, 0.0412, 0.5538, 0.9605, 0.3195, 0.9085],
            ],
            &device,
        )
        .require_grad();

        let mut optimizer = AdanConfig::new()
            .with_beta_1(0.98)
            .with_beta_2(0.92)
            .with_beta_3(0.99)
            .with_epsilon(1e-8)
            .with_weight_decay(0.02)
            .init();

        let grads = linear.forward(x_1).backward();
        let grads = GradientsParams::from_grads(grads, &linear);
        let linear = optimizer.step(LEARNING_RATE, linear, grads);

        let grads = linear.forward(x_2).backward();
        let grads = GradientsParams::from_grads(grads, &linear);
        let linear = optimizer.step(LEARNING_RATE, linear, grads);

        let state_updated = linear.into_record();
        let weights_expected = TensorData::from([
            [
                -0.34034607,
                0.11747075,
                0.38426402,
                0.29999772,
                0.06599136,
                0.04719888,
            ],
            [
                0.0644293,
                -0.031732224,
                -0.37979296,
                0.24165839,
                0.18218218,
                -0.30532277,
            ],
            [
                -0.038910445,
                0.01466812,
                -0.31599957,
                0.2283826,
                -0.29780683,
                0.2929568,
            ],
            [
                -0.3178632,
                -0.24129382,
                -0.39133376,
                -0.31796312,
                -0.09605193,
                0.14255258,
            ],
            [
                0.31026322,
                -0.23771758,
                0.3519465,
                -0.19243571,
                0.35984334,
                -0.049992695,
            ],
            [
                -0.03577819,
                -0.031879753,
                0.10586514,
                0.17213862,
                0.009403733,
                0.36326218,
            ],
        ]);
        let bias_expected = TensorData::from([
            -0.4103378,
            0.06837065,
            -0.116955206,
            0.097558975,
            0.11655137,
            -0.006999196,
        ]);

        let (weight_updated, bias_updated) = (
            state_updated.weight.to_data(),
            state_updated.bias.unwrap().to_data(),
        );

        let tolerance = Tolerance::absolute(1e-5);
        bias_updated.assert_approx_eq::<FT>(&bias_expected, tolerance);
        weight_updated.assert_approx_eq::<FT>(&weights_expected, tolerance);
    }

    #[test]
    fn test_adan_optimizer_no_nan() {
        let linear = given_linear_layer(
            TensorData::from([
                [-0.3206, 0.1374, 0.4043, 0.3200, 0.0859, 0.0671],
                [0.0777, -0.0185, -0.3667, 0.2550, 0.1955, -0.2922],
                [-0.0190, 0.0346, -0.2962, 0.2484, -0.2780, 0.3130],
                [-0.2980, -0.2214, -0.3715, -0.2981, -0.0761, 0.1626],
                [0.3300, -0.2182, 0.3717, -0.1729, 0.3796, -0.0304],
                [-0.0159, -0.0120, 0.1258, 0.1921, 0.0293, 0.3833],
            ]),
            TensorData::from([-0.3905, 0.0884, -0.0970, 0.1176, 0.1366, 0.0130]),
        );

        let x = Tensor::<TestAutodiffBackend, 2>::from_floats(
            [
                [0.8491, 0.2108, 0.8939, 0.4433, 0.5527, 0.2528],
                [0.3270, 0.0412, 0.5538, 0.9605, 0.3195, 0.9085],
            ],
            &Default::default(),
        )
        .require_grad();

        let mut optimizer = AdanConfig::new()
            .with_epsilon(1e-8)
            .with_weight_decay(0.02)
            .init();

        let grads = linear.forward(x.clone()).backward();
        let grads = GradientsParams::from_grads(grads, &linear);
        let linear = optimizer.step(LEARNING_RATE, linear, grads);

        let grads = linear.forward(x).backward();
        let grads = GradientsParams::from_grads(grads, &linear);
        let linear = optimizer.step(LEARNING_RATE, linear, grads);

        let state_updated = linear.into_record();
        assert!(!state_updated.weight.to_data().as_slice::<f32>().unwrap()[0].is_nan());
    }

    fn given_linear_layer(weight: TensorData, bias: TensorData) -> Linear<TestAutodiffBackend> {
        let device = Default::default();
        let record = LinearRecord {
            weight: Param::from_data(weight, &device),
            bias: Some(Param::from_data(bias, &device)),
        };

        LinearConfig::new(6, 6).init(&device).load_record(record)
    }

    fn create_adan() -> OptimizerAdaptor<Adan, Linear<TestAutodiffBackend>, TestAutodiffBackend> {
        let config = AdanConfig::new();
        Adan {
            momentum: AdaptiveNesterovMomentum {
                beta_1: config.beta_1,
                beta_2: config.beta_2,
                beta_3: config.beta_3,
                epsilon: config.epsilon,
            },
            weight_decay: config.weight_decay,
            no_prox: config.no_prox,
        }
        .into()
    }
}
