use crate::{
    self as burn, grad_clipping::GradientClippingConfig, module::ADModule, record::Record,
    LearningRate,
};

use super::{
    decay::{WeightDecay, WeightDecayConfig, WeightDecayState},
    SimpleOptimizer,
};
use crate::config::Config;
use crate::optim::adaptor::OptimizerAdaptor;
use crate::tensor::{backend::ADBackend, Tensor};
use burn_tensor::backend::Backend;

/// Configuration to create the [RMSProp](RMSProp) optimizer.
#[derive(Config)]
pub struct RMSPropConfig {
    /// Smoothing constant.
    #[config(default = 0.99)]
    alpha: f32,
    /// momentum for RMSProp.
    #[config(default = 0.9)]
    momentum: f32,
    /// A value required for numerical stability.
    #[config(default = 1e-5)]
    epsilon: f32,
    /// if True, compute the centered RMSProp, the gradient is normalized by an estimation of its variance
    #[config(default = false)]
    centered: bool,
    /// [Weight decay](WeightDecayConfig) config.
    weight_decay: Option<WeightDecayConfig>,
    /// [Gradient Clipping](GradientClippingConfig) config.
    grad_clipping: Option<GradientClippingConfig>,
}

impl RMSPropConfig {
    /// Initialize RMSProp optimizer.
    ///
    /// # Returns
    ///
    /// Returns an optimizer that can be used to optimize a module.
    pub fn init<B: ADBackend, M: ADModule<B>>(
        &self,
    ) -> OptimizerAdaptor<RMSProp<B::InnerBackend>, M, B> {
        let weight_decay = self.weight_decay.as_ref().map(WeightDecay::new);

        let mut optim = OptimizerAdaptor::from(RMSProp {
            alpha: self.alpha,
            centered: self.centered,
            weight_decay: weight_decay,
            momentum: RMSPropMomentum {
                momentum: self.momentum,
                epsilon: self.epsilon,
            },
        });

        if let Some(config) = &self.grad_clipping {
            optim = optim.with_grad_clipping(config.init());
        }

        optim
    }
}

/// Optimizer that implements stochastic gradient descent with momentum.
/// The optimizer can be configured with [RMSPropConfig](RMSPropConfig).
pub struct RMSProp<B: Backend> {
    alpha: f32,
    // epsilon: f32,
    centered: bool,
    // momentum: Option<Momentum<B>>,
    momentum: RMSPropMomentum,
    weight_decay: Option<WeightDecay<B>>,
}

/// State of [RMSProp](RMSProp)
#[derive(Record, Clone, new)]
pub struct RMSPropState<B: Backend, const D: usize> {
    weight_decay: Option<WeightDecayState<B, D>>,
    square_avg: Option<SquareAvgState<B, D>>,
    centered: Option<CenteredState<B, D>>,
    momentum: Option<RMSPropMomentumState<B, D>>,
}

/// [SquareAvgState](SquareAvgState) is to store and pass optimizer step params.
#[derive(Record, Clone, new)]
pub struct SquareAvgState<B: Backend, const D: usize> {
    square_avg: Tensor<B, D>,
}

impl<B: Backend, const D: usize> SquareAvgState<B, D> {
    /// transform [SquareAvgState] to the next step
    pub fn transform(alpha: f32, grad: Tensor<B, D>, state: Option<Self>) -> (Tensor<B, D>, Self) {
        match state {
            Some(state) => {
                let square_avg = state
                    .square_avg
                    .clone()
                    .mul_scalar(alpha)
                    .add(grad.clone().powf(2.).mul_scalar(1. - alpha));
                (grad, Self { square_avg })
            }
            _ => {
                let square_avg = grad.clone().powf(2.).mul_scalar(1. - alpha);
                (grad, Self { square_avg })
            }
        }
    }

    fn to_device(mut self, device: &B::Device) -> Self {
        self.square_avg = self.square_avg.to_device(device);
        self
    }
}

/// [CenteredState](CenteredState) is to store and pass optimizer step params.
#[derive(Record, Clone, new)]
pub struct CenteredState<B: Backend, const D: usize> {
    grad_avg: Tensor<B, D>,
    avg: Tensor<B, D>,
}

impl<B: Backend, const D: usize> CenteredState<B, D> {
    /// transform [CenteredState] to the next step
    pub fn transform(
        alpha: f32,
        centered: bool,
        grad: Tensor<B, D>,
        centered_state: Option<Self>,
        square_avg_state: Option<SquareAvgState<B, D>>,
    ) -> (Tensor<B, D>, Self, SquareAvgState<B, D>) {
        if centered {
            let grad_avg = match centered_state {
                Some(state) => state
                    .grad_avg
                    .clone()
                    .mul_scalar(alpha)
                    .add(grad.clone().mul_scalar(1. - alpha)),
                _ => grad.clone().mul_scalar(1. - alpha),
            };
            let (avg, square_avg) = match square_avg_state {
                Some(state) => (
                    state.square_avg.clone().sub(grad_avg.clone().powf(2.)),
                    state.square_avg,
                ),
                _ => {
                    // same as init value of SquareAvgState.transform
                    let square_avg = grad.clone().powf(2.).mul_scalar(1. - alpha);
                    (
                        square_avg
                            .clone()
                            .sub(grad_avg.clone().powf(2.).mul_scalar(-1.)),
                        square_avg,
                    )
                }
            };
            (grad, Self { grad_avg, avg }, SquareAvgState { square_avg })
        } else {
            let grad_avg = Tensor::zeros(grad.shape());
            let (avg, square_avg) = match square_avg_state {
                Some(state) => (state.square_avg.clone(), state.square_avg),
                _ => {
                    // same as init value of SquareAvgState.transform
                    let square_avg = grad.clone().powf(2.).mul_scalar(1. - alpha);
                    (square_avg.clone(), square_avg)
                }
            };
            (grad, Self { grad_avg, avg }, SquareAvgState { square_avg })
        }
    }

    fn to_device(mut self, device: &B::Device) -> Self {
        self.grad_avg = self.grad_avg.to_device(device);
        self.avg = self.avg.to_device(device);
        self
    }
}

/// [RMSPropMomentum](RMSPropMomentum) is to store config status for optimizer.
/// (, which is stored in [optimizer](RMSProp) itself and not passed in during `step()` calculation)
pub struct RMSPropMomentum {
    momentum: f32,
    epsilon: f32,
}

impl RMSPropMomentum {
    /// transform [grad](Tensor) and [RMSPropMomentumState] to the next step
    pub fn transform<B: Backend, const D: usize>(
        &self,
        grad: Tensor<B, D>,
        momentum_state: Option<RMSPropMomentumState<B, D>>,
        square_avg_state: Option<SquareAvgState<B, D>>,
    ) -> (
        Tensor<B, D>,
        RMSPropMomentumState<B, D>,
        SquareAvgState<B, D>,
    ) {
        let (grad, square_avg) = match square_avg_state {
            Some(state) => (
                grad.clone()
                    .div(state.square_avg.clone().sqrt().add_scalar(self.epsilon)),
                state.square_avg,
            ),
            _ => {
                let square_avg = Tensor::<B, D>::zeros(grad.shape());
                (
                    grad.clone()
                        .div(square_avg.clone().add_scalar(self.epsilon)),
                    square_avg,
                )
            }
        };

        if self.momentum > 0. {
            let velocity = match momentum_state {
                Some(state) => state
                    .velocity
                    .clone()
                    .mul_scalar(self.momentum)
                    .add(grad.clone()),
                _ => grad.clone(),
            };
            (
                velocity.clone(),
                RMSPropMomentumState { velocity },
                SquareAvgState { square_avg },
            )
        } else {
            match momentum_state {
                Some(state) => (grad.clone(), state, SquareAvgState { square_avg }),
                _ => (
                    grad.clone(),
                    RMSPropMomentumState {
                        velocity: Tensor::zeros(grad.shape()),
                    },
                    SquareAvgState { square_avg },
                ),
            }
        }
    }
}

/// [RMSPropMomentumState](RMSPropMomentumState) is to store and pass optimizer step params.
#[derive(Record, Clone, new)]
pub struct RMSPropMomentumState<B: Backend, const D: usize> {
    velocity: Tensor<B, D>,
}

impl<B: Backend, const D: usize> RMSPropMomentumState<B, D> {
    fn to_device(mut self, device: &B::Device) -> Self {
        self.velocity = self.velocity.to_device(device);
        self
    }
}

impl<B: Backend> SimpleOptimizer<B> for RMSProp<B> {
    type State<const D: usize> = RMSPropState<B, D>;

    fn step<const D: usize>(
        &self,
        lr: LearningRate,
        mut tensor: Tensor<B, D>,
        mut grad: Tensor<B, D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<B, D>, Option<Self::State<D>>) {
        println!("\nparam tensor == {}", tensor);
        println!("\nparam grad == {}", grad);

        // fetch state for params
        let mut state_weight_decay = None;
        let mut state_square_avg = None;
        let mut state_centered = None;
        let mut state_momentum = None;
        if let Some(state) = state {
            state_weight_decay = state.weight_decay;
            state_square_avg = state.square_avg;
            state_centered = state.centered;
            state_momentum = state.momentum;
        }

        // weight_decay transform
        if let Some(weight_decay) = &self.weight_decay {
            let (grad_out, tensor_out, state) =
                weight_decay.transform_temp_fix(grad, tensor, state_weight_decay);
            grad = grad_out;
            tensor = tensor_out;
            state_weight_decay = Some(state);
            println!("\nafter weight_decay tensor=={}", tensor);
            println!("\nafter weight_decay grad=={}", grad);
        }

        // square_avg transform
        let (mut grad, square_avg_out) =
            SquareAvgState::transform(self.alpha, grad, state_square_avg);
        println!("\nafter square_avg grad=={}", grad);
        println!(
            "\nafter square_avg square_avg=={}",
            square_avg_out.square_avg
        );
        state_square_avg = Some(square_avg_out);

        // centered trnsform
        let (grad_out, centered_out, square_avg_out) = CenteredState::transform(
            self.alpha,
            self.centered,
            grad,
            state_centered,
            state_square_avg,
        );
        grad = grad_out;
        state_centered = Some(centered_out);
        state_square_avg = Some(square_avg_out);
        println!("\nafter centered grad=={}", grad);

        // momentum transform
        let (grad, momentum, square_avg) =
            self.momentum
                .transform(grad, state_momentum, state_square_avg);
        println!("\nafter momentum grad=={}", grad);
        println!("\nafter momentum momentum=={}", momentum.velocity);
        let state_square_avg = Some(square_avg);
        let state_momentum = Some(momentum);

        // transition state
        let state = RMSPropState::new(
            state_weight_decay,
            state_square_avg,
            state_centered,
            state_momentum,
        );

        // tensor param transform
        let delta = grad.mul_scalar(lr);
        // (tensor - delta, Some(state))
        let tmp = tensor - delta;
        println!("\nfinal tensor=={}", tmp);
        (tmp, Some(state))
    }

    fn to_device<const D: usize>(
        mut state: Self::State<D>,
        device: &<B as Backend>::Device,
    ) -> Self::State<D> {
        state.weight_decay = state.weight_decay.map(|state| state.to_device(device));
        state.square_avg = state.square_avg.map(|state| state.to_device(device));
        state.centered = state.centered.map(|state| state.to_device(device));
        state.momentum = state.momentum.map(|state| state.to_device(device));
        state
    }
}

#[cfg(test)]
mod tests {
    use burn_tensor::Shape;

    use super::*;
    use crate::module::{Module, Param};
    use crate::optim::{GradientsParams, Optimizer};
    use crate::tensor::{Data, Distribution, Tensor};
    use crate::{nn, TestADBackend, TestBackend};

    const LEARNING_RATE: LearningRate = 0.01;
    const ASSERT_PRECISION: usize = 6;

    #[test]
    fn test_load_state() {
        let layer = nn::LinearConfig::new(20, 20).with_bias(true).init();
        let mut optim = create_rmsprop();
        let loss = layer.forward(create_random_tensor());
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &layer);
        let _layer = optim.step(LEARNING_RATE, layer, grads);

        let record = optim.to_record();
        assert!(!record.is_empty());
    }

    #[test]
    fn test_rmsprop_optimizer_with_numbers_2() {
        let linear = given_linear_layer(
            Data::from([
                [1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1.],
            ]),
            Data::from([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
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

        // let mut optimizer = create_rmsprop();
        let mut optimizer = RMSPropConfig::new()
            .with_alpha(0.99)
            .with_epsilon(1e-8)
            .with_weight_decay(WeightDecayConfig::new(0.05).into())
            .with_momentum(0.9)
            .with_centered(false)
            .init();

        println!("linear is {:?}", linear);
        let grads = linear.forward(x_1).backward();
        let grads = GradientsParams::from_grads(grads, &linear);
        let linear = optimizer.step(LEARNING_RATE, linear, grads);

        println!("linear is {:?}", linear);
        let grads = linear.forward(x_2).backward();
        let grads = GradientsParams::from_grads(grads, &linear);
        let linear = optimizer.step(LEARNING_RATE, linear, grads);

        println!("linear is {:?}", linear);
        let state_updated = linear.into_record();

        let (weight_updated, bias_updated) = (
            state_updated.weight.to_data(),
            state_updated.bias.unwrap().to_data(),
        );

        println!("\nweight_updated\n{:?}", weight_updated);
        println!("\nbias_updated\n{:?}", bias_updated);
    }

    #[test]
    fn test_rmsprop_optimizer_with_numbers() {
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

        // let mut optimizer = create_rmsprop();
        let mut optimizer = RMSPropConfig::new()
            .with_alpha(0.99)
            .with_epsilon(1e-8)
            .with_weight_decay(WeightDecayConfig::new(0.05).into())
            .with_momentum(0.9)
            .with_centered(false)
            .init();

        let grads = linear.forward(x_1).backward();
        let grads = GradientsParams::from_grads(grads, &linear);
        let linear = optimizer.step(LEARNING_RATE, linear, grads);

        let grads = linear.forward(x_2).backward();
        let grads = GradientsParams::from_grads(grads, &linear);
        let linear = optimizer.step(LEARNING_RATE, linear, grads);

        let state_updated = linear.into_record();
        let weights_expected = Data::from([
            [
                -0.576399, -0.076070, 0.147256, 0.060396, -0.165669, -0.189013,
            ],
            [
                -0.178181, -0.231448, -0.623644, -0.004601, -0.056122, -0.548240,
            ],
            [
                -0.274862, -0.178527, -0.553153, -0.011201, -0.529388, 0.056839,
            ],
            [
                -0.553804, -0.433653, -0.628443, -0.557675, -0.327589, -0.093531,
            ],
            [
                0.074068, -0.430464, 0.114660, -0.432481, 0.127890, -0.286493,
            ],
            [
                -0.271762, -0.224970, -0.131209, -0.067498, -0.222241, 0.127126,
            ],
        ]);
        let bias_expected = Data::from([
            -0.651299, -0.172400, -0.357800, -0.143200, -0.124200, -0.247800,
        ]);

        let (weight_updated, bias_updated) = (
            state_updated.weight.to_data(),
            state_updated.bias.unwrap().to_data(),
        );

        println!("\nweight_expected{}\n", weights_expected);
        println!("\nweight_updated\n{}", weight_updated);
        println!("\nbias_updated\n{}", bias_updated);

        bias_updated.assert_approx_eq(&bias_expected, ASSERT_PRECISION);
        weight_updated.assert_approx_eq(&weights_expected, ASSERT_PRECISION);
    }

    fn given_linear_layer(weight: Data<f32, 2>, bias: Data<f32, 1>) -> nn::Linear<TestADBackend> {
        let record = nn::LinearRecord {
            weight: Param::from(Tensor::from_data(weight)),
            bias: Some(Param::from(Tensor::from_data(bias))),
        };

        nn::LinearConfig::new(6, 6).init_with(record)
    }

    fn create_random_tensor() -> Tensor<TestADBackend, 2> {
        Tensor::<TestADBackend, 2>::random(Shape::new([2, 20]), Distribution::Default)
    }

    fn create_rmsprop(
    ) -> OptimizerAdaptor<RMSProp<TestBackend>, nn::Linear<TestADBackend>, TestADBackend> {
        RMSPropConfig {
            alpha: 0.99,
            epsilon: 1e-9,
            centered: false,
            weight_decay: Some(WeightDecayConfig { penalty: 0.05 }),
            momentum: 0.9,
            grad_clipping: None,
            ..RMSPropConfig::new()
        }
        .init()
    }
}
