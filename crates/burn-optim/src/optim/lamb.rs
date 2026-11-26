use burn_core as burn;

use crate::{LearningRate, RecordState, grad_clipping::GradientClippingConfig};
use burn::{
    config::Config,
    tensor::{Device, FloatDType, Tensor},
};

use super::{Optimizer, module_optimizer::ModuleOptimizer};

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float as _;

/// [`Lamb`] configuration.
#[derive(Config, Debug)]
pub struct LambConfig {
    /// Exponential decay rate for the first moment estimates.
    #[config(default = 0.9)]
    beta_1: f32,
    /// Exponential decay rate for the second moment estimates.
    #[config(default = 0.999)]
    beta_2: f32,
    /// A value added to the denominator for numerical stability.
    #[config(default = 1e-6)]
    epsilon: f32,
    /// Weight decay applied to the Adam update before layer-wise adaptation.
    #[config(default = 0.0)]
    weight_decay: f32,
    /// Whether to scale each parameter update by its layer-wise trust ratio.
    #[config(default = true)]
    use_trust_ratio: bool,
    /// Optional gradient clipping configuration.
    grad_clipping: Option<GradientClippingConfig>,
}

/// Layer-wise Adaptive Moments (LAMB) optimizer.
///
/// LAMB applies an Adam-style update, then scales it by the ratio between the parameter norm and
/// update norm. This layer-wise adaptation is useful when training with very large batch sizes.
///
/// See: [Large Batch Optimization for Deep Learning: Training BERT in 76 minutes](https://arxiv.org/abs/1904.00962).
///
/// Configured by [`LambConfig`].
#[derive(Clone)]
pub struct Lamb {
    beta_1: f32,
    beta_2: f32,
    epsilon: f32,
    weight_decay: f32,
    use_trust_ratio: bool,
}

/// LAMB state for a single parameter tensor.
#[derive(RecordState, Clone)]
pub struct LambState<const D: usize> {
    /// The number of optimization steps applied to the parameter.
    pub time: usize,
    /// Exponential moving average of gradients.
    pub moment_1: Tensor<D>,
    /// Exponential moving average of squared gradients.
    pub moment_2: Tensor<D>,
}

impl Optimizer for Lamb {
    type State<const D: usize> = LambState<D>;

    fn step<const D: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<D>,
        grad: Tensor<D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<D>, Option<Self::State<D>>) {
        let factor_1 = 1.0 - self.beta_1;
        let factor_2 = 1.0 - self.beta_2;

        let state = if let Some(mut state) = state {
            state.moment_1 = state
                .moment_1
                .mul_scalar(self.beta_1)
                .add(grad.clone().mul_scalar(factor_1));
            state.moment_2 = state
                .moment_2
                .mul_scalar(self.beta_2)
                .add(grad.square().mul_scalar(factor_2));
            state.time += 1;
            state
        } else {
            LambState {
                time: 1,
                moment_1: grad.clone().mul_scalar(factor_1),
                moment_2: grad.square().mul_scalar(factor_2),
            }
        };

        let time = state.time as i32;
        let moment_1 = state
            .moment_1
            .clone()
            .div_scalar(1.0 - self.beta_1.powi(time));
        let moment_2 = state
            .moment_2
            .clone()
            .div_scalar(1.0 - self.beta_2.powi(time));

        let mut update = moment_1.div(moment_2.sqrt().add_scalar(self.epsilon));
        if self.weight_decay != 0.0 {
            update = update.add(tensor.clone().mul_scalar(self.weight_decay));
        }

        let update = if self.use_trust_ratio {
            let parameter_norm = l2_norm(tensor.clone());
            let update_norm = l2_norm(update.clone());
            let valid_norms = parameter_norm
                .clone()
                .greater_scalar(0.0)
                .bool_and(update_norm.clone().greater_scalar(0.0));

            // Avoid forming 0 / 0 even though the invalid result would be masked out below.
            let min_positive = update
                .dtype()
                .finfo()
                .unwrap_or(FloatDType::F32.finfo())
                .min_positive;
            let ratio = parameter_norm.div(update_norm.clamp_min(min_positive));
            let trust_ratio = ratio.ones_like().mask_where(valid_norms, ratio);

            update.mul(trust_ratio.unsqueeze())
        } else {
            update
        };

        let tensor = tensor - update.mul_scalar(lr);
        (tensor, Some(state))
    }

    fn to_device<const D: usize>(mut state: Self::State<D>, device: &Device) -> Self::State<D> {
        state.moment_1 = state.moment_1.to_device(device);
        state.moment_2 = state.moment_2.to_device(device);
        state
    }
}

impl LambConfig {
    /// Build the per-parameter LAMB optimizer.
    ///
    /// Use [`Self::init`] to construct a whole-module optimizer with the configured gradient
    /// clipping behavior.
    pub fn build(&self) -> Lamb {
        Lamb {
            beta_1: self.beta_1,
            beta_2: self.beta_2,
            epsilon: self.epsilon,
            weight_decay: self.weight_decay,
            use_trust_ratio: self.use_trust_ratio,
        }
    }

    /// Initialize a whole-module LAMB optimizer.
    pub fn init(&self) -> ModuleOptimizer {
        let mut optimizer = ModuleOptimizer::from(self.build());
        if let Some(config) = &self.grad_clipping {
            optimizer = optimizer.with_grad_clipping(config.init());
        }
        optimizer
    }
}

fn l2_norm<const D: usize>(tensor: Tensor<D>) -> Tensor<1> {
    tensor.square().sum().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GradientsParams;
    use burn::{
        module::Param,
        tensor::{TensorData, Tolerance},
    };
    use burn_nn::Linear;

    #[test]
    fn test_lamb_matches_pytorch_reference_for_two_steps() {
        let device = Device::default();
        let optimizer = LambConfig::new()
            .with_beta_1(0.9)
            .with_beta_2(0.999)
            .with_epsilon(1e-6)
            .with_weight_decay(0.1)
            .build();
        let tensor = Tensor::<1>::from_floats([1.0, -2.0, 3.0], &device);

        let (tensor, state) = optimizer.step(
            0.01,
            tensor,
            Tensor::from_floats([0.1, -0.2, 0.3], &device),
            None,
        );
        tensor.to_data().assert_approx_eq::<f32>(
            &TensorData::from([0.9802435, -1.9784473, 2.9766512]),
            Tolerance::absolute(1e-6),
        );

        let (tensor, state) = optimizer.step(
            0.01,
            tensor,
            Tensor::from_floats([-0.4, 0.5, -0.6], &device),
            state,
        );
        tensor.to_data().assert_approx_eq::<f32>(
            &TensorData::from([1.0127187, -1.9956441, 2.9814672]),
            Tolerance::absolute(1e-6),
        );

        let state = state.unwrap();
        assert_eq!(state.time, 2);
        state.moment_1.to_data().assert_approx_eq::<f32>(
            &TensorData::from([-0.031, 0.032, -0.033]),
            Tolerance::absolute(1e-7),
        );
        state.moment_2.to_data().assert_approx_eq::<f32>(
            &TensorData::from([0.00016999, 0.00028996, 0.00044991]),
            Tolerance::absolute(1e-8),
        );
    }

    #[test]
    fn test_lamb_zero_norm_uses_unit_trust_ratio() {
        let device = Device::default();
        let optimizer = LambConfig::new().with_weight_decay(0.1).build();
        let tensor = Tensor::<1>::zeros([2], &device);
        let grad = Tensor::<1>::zeros([2], &device);

        let (tensor, _) = optimizer.step(0.01, tensor, grad, None);

        tensor
            .to_data()
            .assert_eq(&TensorData::from([0.0f32, 0.0]), true);
    }

    #[test]
    fn test_lamb_can_disable_trust_ratio() {
        let device = Device::default();
        let optimizer = LambConfig::new()
            .with_epsilon(1e-6)
            .with_weight_decay(0.1)
            .with_use_trust_ratio(false)
            .build();
        let tensor = Tensor::<1>::from_floats([1.0, -2.0, 3.0], &device);
        let grad = Tensor::<1>::from_floats([0.1, -0.2, 0.3], &device);

        let (tensor, _) = optimizer.step(0.01, tensor, grad, None);

        tensor.to_data().assert_approx_eq::<f32>(
            &TensorData::from([0.9890001, -1.988, 2.987]),
            Tolerance::absolute(1e-6),
        );
    }

    #[test]
    fn test_lamb_state_survives_burnpack_round_trip() {
        let device = Device::default().autodiff();
        let linear = Linear {
            weight: Param::from_data(TensorData::from([[1.0, -2.0], [3.0, -4.0]]), &device),
            bias: Some(Param::from_data(TensorData::from([0.5, -0.5]), &device)),
        };
        let input = Tensor::<2>::from_floats([[0.25, -0.75]], &device).require_grad();
        let mut optimizer = LambConfig::new().with_weight_decay(0.1).init();

        let grads = GradientsParams::from_grads(linear.forward(input.clone()).backward(), &linear);
        let linear = optimizer.step(0.01, linear, grads);
        let bytes = optimizer.into_bytes().unwrap();
        assert!(!bytes.is_empty());

        let mut reloaded = LambConfig::new()
            .with_weight_decay(0.1)
            .init()
            .from_bytes(bytes)
            .unwrap();
        let grads_original =
            GradientsParams::from_grads(linear.forward(input.clone()).backward(), &linear);
        let grads_reloaded = GradientsParams::from_grads(linear.forward(input).backward(), &linear);

        let from_original = optimizer.step(0.01, linear.clone(), grads_original);
        let from_reloaded = reloaded.step(0.01, linear, grads_reloaded);

        from_original
            .weight
            .to_data()
            .assert_approx_eq::<f32>(&from_reloaded.weight.to_data(), Tolerance::absolute(1e-6));
        from_original
            .bias
            .unwrap()
            .to_data()
            .assert_approx_eq::<f32>(
                &from_reloaded.bias.unwrap().to_data(),
                Tolerance::absolute(1e-6),
            );
    }
}
