use crate::{
    self as burn, grad_clipping::GradientClippingConfig, module::ADModule, record::Record,
    LearningRate,
};

use super::{
    decay::{WeightDecay, WeightDecayConfig, WeightDecayState},
    momentum::{self, MomemtumState, Momentum, MomentumConfig},
    Optimizer, SimpleOptimizer,
};
use crate::config::Config;
use crate::optim::adaptor::OptimizerAdaptor;
use crate::tensor::{backend::ADBackend, Tensor};
use burn_tensor::{backend::Backend, ElementConversion};

/// Configuration to create the [RMSProp](RMSProp) optimizer.
#[derive(Config)]
pub struct RMSPropConfig {
    /// Smoothing constant.
    #[config(default = 0.99)]
    alpha: f32,
    /// A value required for numerical stability.
    #[config(default = 1e-5)]
    epsilon: f32,
    /// if True, compute the centered RMSProp, the gradient is normalized by an estimation of its variance
    #[config(default = false)]
    centered: bool,
    /// [Momentum](MomentumConfig) config.
    momentum: Option<MomentumConfig>,
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
        let momentum = self.momentum.as_ref().map(Momentum::new);
        let weight_decay = self.weight_decay.as_ref().map(WeightDecay::new);

        let mut optim = OptimizerAdaptor::from(RMSProp {
            alpha: self.alpha,
            epsilon: self.epsilon,
            centered: self.centered,
            momentum: momentum,
            weight_decay: weight_decay,
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
    epsilon: f32,
    centered: bool,
    momentum: Option<Momentum<B>>,
    weight_decay: Option<WeightDecay<B>>,
}

/// State of [RMSProp](RMSProp)
#[derive(Record, Clone, new)]
pub struct RMSPropState<B: Backend, const D: usize> {
    weight_decay: Option<WeightDecayState<B, D>>,
    momentum: Option<MomemtumState<B, D>>,
    square_avg: Option<Tensor<B, D>>,
    grad_avg: Option<Tensor<B, D>>,
    avg: Option<Tensor<B, D>>,
}

// TODO
pub struct CenteredState<B: Backend, const D: usize> {
    grad_avg: Option<Tensor<B, D>>,
    avg: Option<Tensor<B, D>>,
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
        let mut state_weight_decay = None;
        let mut state_square_avg = None;
        let mut state_grad_avg = None;
        let mut state_avg = None;
        let mut state_momemtum = None;

        if let Some(state) = state {
            state_weight_decay = state.weight_decay;
            state_momemtum = state.momentum;
            state_grad_avg = state.grad_avg;
            state_square_avg = state.square_avg;
            state_avg = state.avg;
        }

        if let Some(weight_decay) = &self.weight_decay {
            let (grad_out, state) = weight_decay.transform(grad, state_weight_decay);
            state_weight_decay = Some(state);
            grad = grad_out;
        }

        let square_avg_out = if let Some(square_avg) = &state_square_avg {
            square_avg
                .clone()
                .mul_scalar(self.alpha)
                .add(grad.clone().powf(2.).mul_scalar(1. - self.alpha))
        } else {
            grad.clone().powf(2.).mul_scalar(1. - self.alpha)
        };
        state_square_avg = Some(square_avg_out);

        if self.centered {
            let grad_avg_out = if let Some(grad_avg) = &state_grad_avg {
                grad_avg
                    .clone()
                    .mul_scalar(self.alpha)
                    .add(grad.clone().mul_scalar(1. - self.alpha))
            } else {
                grad.clone().mul_scalar(1. - self.alpha)
            };
            let avg_out = if let Some(velocity) = &state_square_avg {
                velocity.clone().sub(grad_avg_out.clone().powf(2.))
            } else {
                grad_avg_out.clone().powf(2.).mul_scalar(-1.)
            };
            state_grad_avg = Some(grad_avg_out);
            state_avg = Some(avg_out);
        }

        if let Some(momentum) = &self.momentum {
            // TODO change this with [RMSPropMomentum]
            let (grad_out, state) = momentum.transform(grad, state_momemtum);
            state_momemtum = Some(state);
            grad = grad_out;
        }

        let state = RMSPropState::new(
            state_weight_decay,
            state_momemtum,
            state_grad_avg,
            state_square_avg,
            state_avg,
        );
        let delta = grad.mul_scalar(lr);

        (tensor - delta, Some(state))
    }

    fn to_device<const D: usize>(
        mut state: Self::State<D>,
        device: &<B as Backend>::Device,
    ) -> Self::State<D> {
        state.weight_decay = state.weight_decay.map(|state| state.to_device(device));
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
            momentum: Some(MomentumConfig {
                momentum: 0.9,
                dampening: 0.1,
                ..MomentumConfig::new()
            }),
            grad_clipping: None,
            ..RMSPropConfig::new()
        }
        .init()
    }
}
