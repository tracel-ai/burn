use crate::{
    self as burn, grad_clipping::GradientClippingConfig, module::AutodiffModule, record::Record,
    LearningRate,
};
use std::marker::PhantomData;

use super::SimpleOptimizer;
use crate::config::Config;
use crate::optim::adaptor::OptimizerAdaptor;
use crate::tensor::{backend::AutodiffBackend, Tensor};
use burn_tensor::{backend::Backend, ElementConversion};

/// AdamW configuration.
#[derive(Config)]
pub struct AdamWConfig {
    /// Parameter for AdamW.
    #[config(default = 0.9)]
    beta_1: f32,
    /// Parameter for AdamW.
    #[config(default = 0.999)]
    beta_2: f32,
    /// A value required for numerical stability.
    #[config(default = 1e-5)]
    epsilon: f32,
    /// Weight decay config.
    #[config(default = 1e-4)]
    weight_decay: f32,
    /// [Gradient Clipping](GradientClippingConfig) config.
    grad_clipping: Option<GradientClippingConfig>,
}

/// AdamW optimizer as described in the paper [Decoupled Weight Decay Regularization, Loshchilov and Hutter, 2019](https://arxiv.org/abs/1711.05101).
#[derive(Clone)]
pub struct AdamW<B: Backend> {
    momentum: AdaptiveMomentumW,
    weight_decay: f32,
    _phantom: PhantomData<B>,
}

/// AdamW state.
#[derive(Record, Clone, new)]
pub struct AdamWState<B: Backend, const D: usize> {
    momentum: AdaptiveMomentumWState<B, D>,
}

impl<B: Backend> SimpleOptimizer<B> for AdamW<B> {
    type State<const D: usize> = AdamWState<B, D>;

    /// A single optimization step for any tensor that represents the parameters of a model.
    fn step<const D: usize>(
        &self,
        // Learning rate.
        lr: LearningRate,
        // Any tensor that represents the parameters of a model.
        tensor: Tensor<B, D>,
        // Gradient of the loss w.r.t. the parameters.
        grad: Tensor<B, D>,
        // State of the optimizer.
        state: Option<Self::State<D>>,
    ) -> (Tensor<B, D>, Option<Self::State<D>>) {
        let tensor_updated = tensor.clone() - tensor.mul_scalar(lr).mul_scalar(self.weight_decay);

        let (raw_delta, momentum_state) = self.momentum.transform(grad, state.map(|s| s.momentum));

        let state = AdamWState {
            momentum: momentum_state,
        };

        (tensor_updated - raw_delta.mul_scalar(lr), Some(state))
    }

    fn to_device<const D: usize>(
        mut state: Self::State<D>,
        device: &<B as Backend>::Device,
    ) -> Self::State<D> {
        state.momentum = state.momentum.to_device(device);
        state
    }
}

impl AdamWConfig {
    /// Initialize AdamW optimizer.
    ///
    /// # Returns
    ///
    /// Returns an optimizer that can be used to optimize a module.
    pub fn init<B: AutodiffBackend, M: AutodiffModule<B>>(
        &self,
    ) -> OptimizerAdaptor<AdamW<B::InnerBackend>, M, B> {
        let optim = AdamW {
            momentum: AdaptiveMomentumW {
                beta_1: self.beta_1,
                beta_2: self.beta_2,
                epsilon: self.epsilon,
            },
            weight_decay: self.weight_decay,
            _phantom: Default::default(),
        };

        let mut optim = OptimizerAdaptor::from(optim);
        if let Some(config) = &self.grad_clipping {
            optim = optim.with_grad_clipping(config.init());
        }
        optim
    }
}

/// Adaptive momentum state.
#[derive(Record, new, Clone)]
pub struct AdaptiveMomentumWState<B: Backend, const D: usize> {
    time: usize,
    moment_1: Tensor<B, D>,
    moment_2: Tensor<B, D>,
}

#[derive(Clone)]
struct AdaptiveMomentumW {
    beta_1: f32,
    beta_2: f32,
    epsilon: f32,
}

impl AdaptiveMomentumW {
    pub fn transform<B: Backend, const D: usize>(
        &self,
        grad: Tensor<B, D>,
        state: Option<AdaptiveMomentumWState<B, D>>,
    ) -> (Tensor<B, D>, AdaptiveMomentumWState<B, D>) {
        let state = if let Some(mut state) = state {
            // Update first moment estimate.
            let factor = 1.0 - self.beta_1;
            state.moment_1 = state
                .moment_1
                .mul_scalar(self.beta_1)
                .add(grad.clone().mul_scalar(factor));

            // Update second moment estimate.
            let factor = 1.0 - self.beta_2;
            state.moment_2 = state
                .moment_2
                .mul_scalar(self.beta_2)
                .add(grad.powf_scalar(2.0).mul_scalar(factor));

            // Update time.
            state.time += 1;

            state
        } else {
            // Initialize first moment estimate.
            let factor = 1.0 - self.beta_1;
            let moment_1 = grad.clone().mul_scalar(factor);

            // Initialize second moment estimate.
            let factor = 1.0 - self.beta_2;
            let moment_2 = grad.powf_scalar(2.0).mul_scalar(factor);

            AdaptiveMomentumWState::new(1, moment_1, moment_2)
        };

        let time: i32 = (state.time as i32).elem();

        // Compute bias-corrected first and second moment estimates.
        let moment_1_corrected = state
            .moment_1
            .clone()
            .div_scalar(1f32 - self.beta_1.powi(time));

        let moment_2_corrected = state
            .moment_2
            .clone()
            .div_scalar(1f32 - self.beta_2.powi(time));

        // Compute update delta. This still needs to be scaled by the learning rate.
        let update_delta =
            moment_1_corrected.div(moment_2_corrected.sqrt().add_scalar(self.epsilon));

        (
            update_delta,
            AdaptiveMomentumWState::new(state.time, state.moment_1, state.moment_2),
        )
    }
}

impl<B: Backend, const D: usize> AdaptiveMomentumWState<B, D> {
    /// Move state to device.
    ///
    /// # Arguments
    ///
    /// * `device` - Device to move state to.
    ///
    /// # Returns
    ///
    /// Returns state moved to device.
    pub fn to_device(mut self, device: &B::Device) -> Self {
        self.moment_1 = self.moment_1.to_device(device);
        self.moment_2 = self.moment_2.to_device(device);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::module::{Module, Param};
    use crate::optim::{GradientsParams, Optimizer};
    use crate::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
    use crate::tensor::{Data, Distribution, Tensor};
    use crate::{nn, TestAutodiffBackend, TestBackend};
    use tempfile::TempDir;

    const LEARNING_RATE: LearningRate = 0.01;

    #[test]
    fn test_adamw_optimizer_save_load_state() {
        let device = Default::default();
        let linear = nn::LinearConfig::new(6, 6).init(&device);
        let x = Tensor::<TestAutodiffBackend, 2>::random([2, 6], Distribution::Default, &device);
        let mut optimizer = create_adamw();
        let grads = linear.forward(x).backward();
        let grads = GradientsParams::from_grads(grads, &linear);
        let _linear = optimizer.step(LEARNING_RATE, linear, grads);
        let temp_dir = TempDir::new().unwrap();
        BinFileRecorder::<FullPrecisionSettings>::default()
            .record(
                optimizer.to_record(),
                temp_dir.path().join("test_optim_adamw"),
            )
            .unwrap();

        let state_optim_before = optimizer.to_record();
        let state_optim_before_copy = optimizer.to_record();
        let optimizer = create_adamw();
        let optimizer = optimizer.load_record(state_optim_before_copy);
        let state_optim_after = optimizer.to_record();

        assert_eq!(state_optim_before.len(), state_optim_after.len());
    }

    const ASSERT_PRECISION: usize = 2;

    #[test]
    fn test_adamw_optimizer_with_numbers() {
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
        let device = Default::default();
        let x_1 = Tensor::from_floats(
            [
                [0.6294, 0.0940, 0.8176, 0.8824, 0.5228, 0.4310],
                [0.7152, 0.9559, 0.7893, 0.5684, 0.5939, 0.8883],
            ],
            &device,
        )
        .require_grad();
        let x_2 = Tensor::from_floats(
            [
                [0.8491, 0.2108, 0.8939, 0.4433, 0.5527, 0.2528],
                [0.3270, 0.0412, 0.5538, 0.9605, 0.3195, 0.9085],
            ],
            &device,
        )
        .require_grad();

        let mut optimizer = AdamWConfig::new()
            .with_epsilon(1e-8)
            .with_beta_1(0.9)
            .with_beta_2(0.999)
            .with_weight_decay(0.5)
            .init();

        let grads = linear.forward(x_1).backward();
        let grads = GradientsParams::from_grads(grads, &linear);
        let linear = optimizer.step(LEARNING_RATE, linear, grads);

        let grads = linear.forward(x_2).backward();
        let grads = GradientsParams::from_grads(grads, &linear);
        let linear = optimizer.step(LEARNING_RATE, linear, grads);

        let state_updated = linear.into_record();
        let weights_expected = Data::from([
            [-0.337295, 0.117827, 0.380358, 0.296868, 0.065232, 0.046534],
            [
                0.057032, -0.036518, -0.382951, 0.232516, 0.173738, -0.309182,
            ],
            [
                -0.038703, 0.016052, -0.313155, 0.225982, -0.295039, 0.289981,
            ],
            [
                -0.314920, -0.237394, -0.387704, -0.315067, -0.095153, 0.141081,
            ],
            [
                0.306815, -0.234226, 0.348083, -0.191115, 0.356002, -0.049993,
            ],
            [-0.035634, -0.030083, 0.104636, 0.170244, 0.009196, 0.359580],
        ]);
        let bias_expected = Data::from([
            -0.406555, 0.067568, -0.115982, 0.096477, 0.115287, -0.007080,
        ]);

        let (weight_updated, bias_updated) = (
            state_updated.weight.to_data(),
            state_updated.bias.unwrap().to_data(),
        );

        bias_updated.assert_approx_eq(&bias_expected, ASSERT_PRECISION);
        weight_updated.assert_approx_eq(&weights_expected, ASSERT_PRECISION);
    }

    #[test]
    fn test_adam_optimizer_no_nan() {
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

        let x = Tensor::from_floats(
            [
                [0.8491, 0.2108, 0.8939, 0.4433, 0.5527, 0.2528],
                [0.3270, 0.0412, 0.5538, 0.9605, 0.3195, 0.9085],
            ],
            &Default::default(),
        )
        .require_grad();

        let mut optimizer = AdamWConfig::new()
            .with_epsilon(1e-8)
            .with_beta_1(0.9)
            .with_beta_2(0.999)
            .with_weight_decay(0.5)
            .init();

        let grads = linear.forward(x.clone()).backward();
        let grads = GradientsParams::from_grads(grads, &linear);
        let linear = optimizer.step(LEARNING_RATE, linear, grads);

        let grads = linear.forward(x).backward();
        let grads = GradientsParams::from_grads(grads, &linear);
        let linear = optimizer.step(LEARNING_RATE, linear, grads);

        let state_updated = linear.into_record();
        assert!(!state_updated.weight.to_data().value[0].is_nan());
    }

    fn given_linear_layer(
        weight: Data<f32, 2>,
        bias: Data<f32, 1>,
    ) -> nn::Linear<TestAutodiffBackend> {
        let device = Default::default();
        let record = nn::LinearRecord {
            weight: Param::from_data(weight, &device),
            bias: Some(Param::from_data(bias, &device)),
        };

        nn::LinearConfig::new(6, 6)
            .init(&device)
            .load_record(record)
    }

    fn create_adamw(
    ) -> OptimizerAdaptor<AdamW<TestBackend>, nn::Linear<TestAutodiffBackend>, TestAutodiffBackend>
    {
        let config = AdamWConfig::new();
        AdamW {
            momentum: AdaptiveMomentumW {
                beta_1: config.beta_1,
                beta_2: config.beta_2,
                epsilon: config.epsilon,
            },
            weight_decay: config.weight_decay,
            _phantom: Default::default(),
        }
        .into()
    }
}
