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
        tensor: Tensor<B, D>,
        mut grad: Tensor<B, D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<B, D>, Option<Self::State<D>>) {
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
            let (grad_out, state) = weight_decay.transform(grad, state_weight_decay);
            state_weight_decay = Some(state);
            grad = grad_out;
        }

        // square_avg transform
        let (mut grad, square_avg_out) =
            SquareAvgState::transform(self.alpha, grad, state_square_avg);
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

        // momentum transform
        let (grad, momentum, square_avg) =
            self.momentum
                .transform(grad, state_momentum, state_square_avg);
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
        (tensor - delta, Some(state))
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
    use burn_tensor::{Distribution, Shape};

    use crate::{
        nn::{Linear, LinearConfig},
        optim::GradientsParams,
        TestADBackend, TestBackend,
    };

    use crate::optim::base::Optimizer;

    use super::*;

    const LEARNING_RATE: LearningRate = 0.02;

    #[test]
    fn test_load_state() {
        let layer = LinearConfig::new(20, 20).with_bias(true).init();
        let mut optim = create_rmsprop();
        let loss = layer.forward(create_random_tensor());
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &layer);
        let _layer = optim.step(LEARNING_RATE, layer, grads);

        let record = optim.to_record();
        assert!(!record.is_empty());
    }

    fn create_random_tensor() -> Tensor<TestADBackend, 2> {
        Tensor::<TestADBackend, 2>::random(Shape::new([2, 20]), Distribution::Default)
    }

    fn create_rmsprop(
    ) -> OptimizerAdaptor<RMSProp<TestBackend>, Linear<TestADBackend>, TestADBackend> {
        RMSPropConfig {
            weight_decay: Some(WeightDecayConfig { penalty: 0.05 }),
            momentum: 0.9,
            grad_clipping: None,
            ..RMSPropConfig::new()
        }
        .init()
    }
}
