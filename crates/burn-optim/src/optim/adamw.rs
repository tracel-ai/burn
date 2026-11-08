use burn_core as burn;

use crate::RecordState;
use burn::config::Config;
use burn::tensor::Device;
use burn::tensor::Tensor;

use super::{AdaptiveMomentumState, Optimizer, module_optimizer::ModuleOptimizer};
use crate::{LearningRate, grad_clipping::GradientClippingConfig};

#[cfg(cubecl_backend)]
mod cube;

#[cfg(all(test, cubecl_backend))]
mod test_support;

#[cfg(all(test, cubecl_backend))]
fn step_guard() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|poison| poison.into_inner())
}

/// Without a CubeCL runtime the fused execution is unavailable, so every step falls back.
#[cfg(not(cubecl_backend))]
mod cube {
    use burn_core::tensor::Tensor;

    use super::{AdamWState, AdaptiveMomentumW, StepInputs};
    use crate::LearningRate;

    #[allow(clippy::result_large_err)]
    pub(super) fn try_step<const D: usize>(
        _lr: LearningRate,
        tensor: Tensor<D>,
        grad: Tensor<D>,
        state: Option<AdamWState<D>>,
        _momentum: &AdaptiveMomentumW,
        _weight_decay: f32,
        _cautious_weight_decay: bool,
    ) -> Result<(Tensor<D>, Option<AdamWState<D>>), StepInputs<D>> {
        Err((tensor, grad, state))
    }
}

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float as _;

/// How [`AdamW`] applies its per-parameter update.
///
/// The enum is `#[non_exhaustive]`: new execution strategies can be added without breaking
/// existing matches.
#[derive(Config, Debug)]
#[non_exhaustive]
pub enum AdamWExecutionConfig {
    /// Build the update from tensor operations, one step at a time.
    ///
    /// This is the default and works on every backend.
    Reference,
    /// Fuse each parameter update, vectorizing eligible CPU buffers.
    ///
    /// Large contiguous buffers with a partial vector use an additional scalar tail launch.
    ///
    /// Available for `F32` tensors on raw CubeCL backends, built with one of `burn-optim`'s
    /// CubeCL runtime features (`cpu`, `cuda`, `rocm`, `wgpu`, ...). Every unsupported case,
    /// including kernel fusion and tensors still in an autodiff context, falls back to
    /// [`Reference`](Self::Reference). The module optimizer strips the autodiff context itself;
    /// a direct [`Optimizer::step`](crate::Optimizer::step) call on a tracked tensor falls back.
    Fused,
}

/// [`AdamW`] Configuration.
#[derive(Config, Debug)]
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

    /// Cautious weight decay config.
    ///
    /// See: <https://arxiv.org/abs/2510.12402>
    #[config(default = false)]
    cautious_weight_decay: bool,

    /// Whether to use AMSGrad algorithm
    #[config(default = false)]
    amsgrad: bool,
    /// [Gradient Clipping](GradientClippingConfig) config.
    grad_clipping: Option<GradientClippingConfig>,

    /// How the per-parameter update is executed.
    ///
    /// `None` loads as [`AdamWExecutionConfig::Reference`]. Configurations written before the
    /// field existed deserialize to `None`.
    execution: Option<AdamWExecutionConfig>,
}

/// AdamW optimizer.
///
/// See:
/// - [Decoupled Weight Decay Regularization, Loshchilov and Hutter, 2019](https://arxiv.org/abs/1711.05101).
/// - [Cautious Weight Decay, 2025](https://arxiv.org/abs/2510.12402)
/// - [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ)
///
/// Configured by [`AdamWConfig`].
#[derive(Clone)]
pub struct AdamW {
    momentum: AdaptiveMomentumW,
    weight_decay: f32,
    cautious_weight_decay: bool,
    execution: AdamWExecutionConfig,
}

/// AdamW state.
#[derive(RecordState, Clone, new)]
pub struct AdamWState<const D: usize> {
    /// Th current adaptive momentum state.
    pub momentum: AdaptiveMomentumState<D>,
}

impl Optimizer for AdamW {
    type State<const D: usize> = AdamWState<D>;

    /// A single optimization step for any tensor that represents the parameters of a model.
    fn step<const D: usize>(
        &self,
        // Learning rate.
        lr: LearningRate,
        // Any tensor that represents the parameters of a model.
        tensor: Tensor<D>,
        // Gradient of the loss w.r.t. the parameters.
        grad: Tensor<D>,
        // State of the optimizer.
        state: Option<Self::State<D>>,
    ) -> (Tensor<D>, Option<Self::State<D>>) {
        match &self.execution {
            AdamWExecutionConfig::Reference => self.reference_step(lr, tensor, grad, state),
            AdamWExecutionConfig::Fused => match cube::try_step(
                lr,
                tensor,
                grad,
                state,
                &self.momentum,
                self.weight_decay,
                self.cautious_weight_decay,
            ) {
                Ok(updated) => updated,
                Err((tensor, grad, state)) => self.reference_step(lr, tensor, grad, state),
            },
        }
    }

    fn to_device<const D: usize>(mut state: Self::State<D>, device: &Device) -> Self::State<D> {
        state.momentum = state.momentum.to_device(device);
        state
    }
}

impl AdamW {
    /// The reference implementation: the update is assembled from tensor operations.
    fn reference_step<const D: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<D>,
        grad: Tensor<D>,
        state: Option<AdamWState<D>>,
    ) -> (Tensor<D>, Option<AdamWState<D>>) {
        let (raw_delta, momentum_state) = self.momentum.transform(grad, state.map(|s| s.momentum));

        let decay_rate = lr * (self.weight_decay as f64);

        let decayed_tensor = if decay_rate == 0.0 {
            tensor.clone()
        } else if self.cautious_weight_decay {
            // Cautious weight decay.
            // See: https://arxiv.org/abs/2510.12402
            let tensor_pos = tensor.clone().greater_equal_scalar(0.0);
            let grad_pos = momentum_state.moment_1.clone().greater_equal_scalar(0.0);
            let differ = tensor_pos.not_equal(grad_pos);

            // Zero out the decay where the decay is counter to the update direction.
            tensor.clone() - tensor.mul_scalar(decay_rate).mask_fill(differ, 0.0)
        } else {
            tensor.clone().mul_scalar(1.0 - decay_rate)
        };

        let tensor_updated = decayed_tensor - raw_delta.mul_scalar(lr);

        let state = AdamWState {
            momentum: momentum_state,
        };

        (tensor_updated, Some(state))
    }
}

/// The inputs of a step that fell back to the reference implementation, unconsumed.
type StepInputs<const D: usize> = (Tensor<D>, Tensor<D>, Option<AdamWState<D>>);

impl AdamWConfig {
    /// Build an [`AdamW`] from the config.
    ///
    /// The bare optimizer, which
    /// [`ModuleOptimizer::with_group`](crate::ModuleOptimizer::with_group) takes to
    /// optimize one parameter group. [`init`](Self::init) is the whole-module
    /// counterpart, and the only one that applies the configured gradient clipping.
    pub fn build(&self) -> AdamW {
        AdamW {
            momentum: AdaptiveMomentumW {
                beta_1: self.beta_1,
                beta_2: self.beta_2,
                epsilon: self.epsilon,
                amsgrad: self.amsgrad,
            },
            weight_decay: self.weight_decay,
            cautious_weight_decay: self.cautious_weight_decay,
            execution: self
                .execution
                .clone()
                .unwrap_or(AdamWExecutionConfig::Reference),
        }
    }

    /// Initialize AdamW optimizer.
    ///
    /// # Returns
    ///
    /// Returns an optimizer that can be used to optimize a module.
    pub fn init(&self) -> ModuleOptimizer {
        let mut optim = ModuleOptimizer::from(self.build());
        if let Some(config) = &self.grad_clipping {
            optim = optim.with_grad_clipping(config.init());
        }
        optim
    }
}

#[derive(Clone)]
struct AdaptiveMomentumW {
    beta_1: f32,
    beta_2: f32,
    epsilon: f32,
    amsgrad: bool,
}

impl AdaptiveMomentumW {
    pub fn transform<const D: usize>(
        &self,
        grad: Tensor<D>,
        state: Option<AdaptiveMomentumState<D>>,
    ) -> (Tensor<D>, AdaptiveMomentumState<D>) {
        let factor_1 = 1.0 - self.beta_1;
        let factor_2 = 1.0 - self.beta_2;

        let state = if let Some(mut state) = state {
            // Update first moment estimate.
            state.moment_1 = state
                .moment_1
                .mul_scalar(self.beta_1)
                .add(grad.clone().mul_scalar(factor_1));

            // Update second moment estimate.
            state.moment_2 = state
                .moment_2
                .mul_scalar(self.beta_2)
                .add(grad.square().mul_scalar(factor_2));

            if self.amsgrad {
                let max_v = state
                    .max_moment_2
                    .take()
                    .unwrap_or_else(|| state.moment_2.clone());
                state.max_moment_2 = Some(max_v.max_pair(state.moment_2.clone()));
            }

            // Update time.
            state.time += 1;

            state
        } else {
            // Initialize first moment estimate.
            let moment_1 = grad.clone().mul_scalar(factor_1);

            // Initialize second moment estimate.
            let moment_2 = grad.square().mul_scalar(factor_2);
            let max_moment_2 = self.amsgrad.then(|| moment_2.clone());
            AdaptiveMomentumState {
                time: 1,
                moment_1,
                moment_2,
                max_moment_2,
            }
        };

        let time: i32 = state.time as i32;

        // Compute bias-corrected first and second moment estimates.
        let moment_1_corrected = state
            .moment_1
            .clone()
            .div_scalar(1f32 - self.beta_1.powi(time));

        let v_to_use = if self.amsgrad {
            state.max_moment_2.as_ref().unwrap_or(&state.moment_2)
        } else {
            &state.moment_2
        };

        let moment_2_corrected = v_to_use.clone().div_scalar(1f32 - self.beta_2.powi(time));

        let update_delta =
            moment_1_corrected.div(moment_2_corrected.sqrt().add_scalar(self.epsilon));

        (update_delta, state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GradientsParams;
    use crate::optim::test_utils::assert_optimizer_resume;
    use burn::module::Param;
    use burn::tensor::Tolerance;
    use burn::tensor::{Tensor, TensorData};
    use burn_nn::{Linear, LinearConfig};

    type FT = f32;

    const LEARNING_RATE: LearningRate = 0.01;

    #[test]
    fn test_adamw_optimizer_save_load_state() {
        let device = Device::default().autodiff();
        let linear = LinearConfig::new(6, 6).init(&device);
        for amsgrad in [false, true] {
            assert_optimizer_resume(
                || AdamWConfig::new().with_amsgrad(amsgrad).init(),
                linear.clone(),
                LEARNING_RATE,
            );
        }
    }
    #[test]
    fn test_adamw_optimizer_with_amsgrad_50_steps() {
        let device = Device::default().autodiff();
        let mut linear = given_linear_layer(
            TensorData::from([
                [-0.3206, 0.1374, 0.4043, 0.3200, 0.0859, 0.0671],
                [0.0777, -0.0185, -0.3667, 0.2550, 0.1955, -0.2922],
                [-0.0190, 0.0346, -0.2962, 0.2484, -0.2780, 0.3130],
                [-0.2980, -0.2214, -0.3715, -0.2981, -0.0761, 0.1626],
                [0.3300, -0.2182, 0.3717, -0.1729, 0.3796, -0.0304],
                [-0.0159, -0.0120, 0.1258, 0.1921, 0.0293, 0.3833],
            ]),
            TensorData::from([-0.3905, 0.0884, -0.0970, 0.1176, 0.1366, 0.0130]),
            &device,
        );

        let mut optimizer = AdamWConfig::new()
            .with_epsilon(1e-8)
            .with_beta_1(0.9)
            .with_beta_2(0.999)
            .with_amsgrad(true)
            .with_weight_decay(0.5)
            .init();

        for i in 1..=50 {
            let x = Tensor::<2>::ones([2, 6], &device)
                .mul_scalar(i as f32 * 0.1)
                .require_grad();

            let grads = linear.forward(x).backward();
            let grads = GradientsParams::from_grads(grads, &linear);
            linear = optimizer.step(LEARNING_RATE, linear, grads);
        }

        let state_updated = linear;
        let weight_updated = state_updated.weight.to_data();
        let bias_updated = state_updated.bias.unwrap().to_data();

        let weights_expected = TensorData::from([
            [
                -0.7822558283805847,
                -0.42578864097595215,
                -0.21805696189403534,
                -0.28366872668266296,
                -0.46587175130844116,
                -0.4805040955543518,
            ],
            [
                -0.4722539782524109,
                -0.5471276640892029,
                -0.8181359767913818,
                -0.33425918221473694,
                -0.3805687427520752,
                -0.7601516842842102,
            ],
            [
                -0.5475167632102966,
                -0.5057991743087769,
                -0.763265073299408,
                -0.3393959403038025,
                -0.7490996718406677,
                -0.28911691904067993,
            ],
            [
                -0.7646660208702087,
                -0.7050473093986511,
                -0.8218720555305481,
                -0.7647438049316406,
                -0.5919585227966309,
                -0.40617525577545166,
            ],
            [
                -0.27588561177253723,
                -0.7025567889213562,
                -0.24343004822731018,
                -0.6672990918159485,
                -0.23728127777576447,
                -0.556389570236206,
            ],
            [
                -0.5451040267944336,
                -0.5420684814453125,
                -0.4348171353340149,
                -0.3832150399684906,
                -0.5099242925643921,
                -0.23440153896808624,
            ],
        ]);
        let bias_expected = TensorData::from([
            -0.7473056316375732,
            -0.3745720386505127,
            -0.5188710689544678,
            -0.35184532403945923,
            -0.33705732226371765,
            -0.4332566559314728,
        ]);

        let tolerance = Tolerance::absolute(1e-5);
        weight_updated.assert_approx_eq::<FT>(&weights_expected, tolerance);
        bias_updated.assert_approx_eq::<FT>(&bias_expected, tolerance);
    }
    #[test]
    fn test_adamw_optimizer_with_numbers() {
        let device = Device::default().autodiff();
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
            &device,
        );
        let x_1 = Tensor::<2>::from_floats(
            [
                [0.6294, 0.0940, 0.8176, 0.8824, 0.5228, 0.4310],
                [0.7152, 0.9559, 0.7893, 0.5684, 0.5939, 0.8883],
            ],
            &device,
        )
        .require_grad();
        let x_2 = Tensor::<2>::from_floats(
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

        let state_updated = linear;
        let weights_expected = TensorData::from([
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
        let bias_expected = TensorData::from([
            -0.406555, 0.067568, -0.115982, 0.096477, 0.115287, -0.007080,
        ]);

        let (weight_updated, bias_updated) = (
            state_updated.weight.to_data(),
            state_updated.bias.unwrap().to_data(),
        );

        let tolerance = Tolerance::absolute(1e-2);
        bias_updated.assert_approx_eq::<FT>(&bias_expected, tolerance);
        weight_updated.assert_approx_eq::<FT>(&weights_expected, tolerance);
    }

    #[test]
    fn test_adamw_optimizer_with_numbers_cautious() {
        let device = Device::default().autodiff();
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
            &device,
        );
        let x_1 = Tensor::<2>::from_floats(
            [
                [0.6294, 0.0940, 0.8176, 0.8824, 0.5228, 0.4310],
                [0.7152, 0.9559, 0.7893, 0.5684, 0.5939, 0.8883],
            ],
            &device,
        )
        .require_grad();
        let x_2 = Tensor::<2>::from_floats(
            [
                [0.8491, 0.2108, 0.8939, 0.4433, 0.5527, 0.2528],
                [0.3270, 0.0412, 0.5538, 0.9605, 0.3195, -0.9085],
            ],
            &device,
        )
        .require_grad();

        let mut optimizer = AdamWConfig::new()
            .with_cautious_weight_decay(true)
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

        let state_updated = linear;
        let weights_expected = TensorData::from([
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
            [
                -0.035634, -0.030083, 0.104636, 0.170244, 0.009196, 0.37061332,
            ],
        ]);
        let bias_expected = TensorData::from([
            -0.406555, 0.067568, -0.115982, 0.096477, 0.115287, -0.007080,
        ]);

        let (weight_updated, bias_updated) = (
            state_updated.weight.to_data(),
            state_updated.bias.unwrap().to_data(),
        );

        let tolerance = Tolerance::absolute(1e-2);
        bias_updated.assert_approx_eq::<FT>(&bias_expected, tolerance);
        weight_updated.assert_approx_eq::<FT>(&weights_expected, tolerance);
    }

    #[test]
    fn test_adam_optimizer_no_nan() {
        let device = Device::default().autodiff();
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
            &device,
        );

        let x = Tensor::<2>::from_floats(
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

        let grads = linear.forward(x.clone()).backward();
        let grads = GradientsParams::from_grads(grads, &linear);
        let linear = optimizer.step(LEARNING_RATE, linear, grads);

        let grads = linear.forward(x).backward();
        let grads = GradientsParams::from_grads(grads, &linear);
        let linear = optimizer.step(LEARNING_RATE, linear, grads);

        let state_updated = linear;
        assert!(!state_updated.weight.to_data().as_slice::<f32>().unwrap()[0].is_nan());
    }

    #[test]
    fn config_without_execution_loads_as_reference() {
        // A configuration written before `execution` existed: the missing field must load.
        let json = r#"{
            "beta_1": 0.9,
            "beta_2": 0.999,
            "epsilon": 0.00001,
            "weight_decay": 0.0001,
            "cautious_weight_decay": false,
            "amsgrad": false,
            "grad_clipping": null
        }"#;
        let config = AdamWConfig::load_binary(json.as_bytes()).unwrap();
        assert!(matches!(
            config.build().execution,
            AdamWExecutionConfig::Reference
        ));
    }

    #[test]
    fn execution_round_trips_through_json_and_binary() {
        for execution in [AdamWExecutionConfig::Reference, AdamWExecutionConfig::Fused] {
            let expected = core::mem::discriminant(&execution);
            let config = AdamWConfig::new().with_execution(Some(execution));
            let json = burn::config::config_to_json(&config);
            let loaded = AdamWConfig::load_binary(json.as_bytes()).unwrap();
            assert_eq!(
                core::mem::discriminant(loaded.execution.as_ref().unwrap()),
                expected
            );
        }
    }

    #[cfg(cubecl_backend)]
    mod fused {
        use super::super::test_support::KernelCounter;
        use super::*;
        use crate::optim::AdaptiveMomentumState;
        use crate::optim::test_utils::assert_optimizer_resume_between;
        use burn::tensor::DType;
        use burn_cubecl::kernel::optim::{AdamWHostScalars, AdamWInputs, try_adamw};

        use super::super::step_guard;

        fn tolerance() -> Tolerance<f32> {
            // Four epsilons of absolute slack and eight of relative: enough for different FMA
            // contraction, division and sqrt instructions, not enough to hide a formula change.
            Tolerance::rel_abs(8.0 * f32::EPSILON, 4.0 * f32::EPSILON)
        }

        fn values(len: usize, scale: f32) -> Vec<f32> {
            (0..len).map(|i| (i as f32 * 0.37).sin() * scale).collect()
        }

        fn tensor<const D: usize>(values: &[f32], shape: [usize; D], device: &Device) -> Tensor<D> {
            Tensor::from_data(TensorData::new(values.to_vec(), shape), device)
        }

        fn fused_config(
            amsgrad: bool,
            cautious_weight_decay: bool,
            weight_decay: f32,
        ) -> AdamWConfig {
            AdamWConfig::new()
                .with_epsilon(1e-8)
                .with_weight_decay(weight_decay)
                .with_amsgrad(amsgrad)
                .with_cautious_weight_decay(cautious_weight_decay)
                .with_execution(Some(AdamWExecutionConfig::Fused))
        }

        /// Run `f` and assert it submitted exactly one AdamW kernel. Fusion builds submit none.
        fn kernel_count(bindings: u32) -> usize {
            if burn_cubecl::FUSED {
                return 0;
            }
            let device = Device::default();
            match device.as_dispatch() {
                burn::backend::DispatchDevice::Cube(device) => {
                    usize::from(device.client().properties().hardware.max_bindings >= bindings)
                }
                #[allow(unreachable_patterns)]
                _ => 0,
            }
        }

        fn one_launch<R>(bindings: u32, f: impl FnOnce() -> R) -> R {
            let (launches, _observation) = KernelCounter::start();
            let before = launches.get();
            let result = f();
            let expected = kernel_count(bindings);
            assert_eq!(
                launches.get() - before,
                expected,
                "the fused step did not reach the kernel"
            );
            result
        }

        /// Run `f` and assert it stayed on the reference path.
        fn no_launch<R>(f: impl FnOnce() -> R) -> R {
            let (launches, _observation) = KernelCounter::start();
            let before = launches.get();
            let result = f();
            assert_eq!(
                launches.get(),
                before,
                "an unsupported step must not reach the kernel"
            );
            result
        }

        fn allocation<const D: usize>(tensor: &Tensor<D>) -> usize {
            let primitive = tensor
                .clone()
                .try_into_primitive::<burn_cubecl::Cube>()
                .unwrap();
            let primitive = (&primitive as &dyn core::any::Any)
                .downcast_ref::<burn_cubecl::tensor::CubeTensor>()
                .expect("allocation assertions require raw CubeCL");
            primitive.handle.memory.descriptor().id.value
        }

        fn output_allocations<const D: usize>(
            param: &Tensor<D>,
            state: &AdamWState<D>,
        ) -> Vec<usize> {
            let mut ids = vec![
                allocation(param),
                allocation(&state.momentum.moment_1),
                allocation(&state.momentum.moment_2),
            ];
            if let Some(max) = &state.momentum.max_moment_2 {
                ids.push(allocation(max));
            }
            ids
        }

        #[test]
        fn fused_step_uses_the_client_stream_when_inputs_were_created_elsewhere() {
            use burn::tensor::StreamId;

            let _guard = step_guard();
            if kernel_count(6) == 0 {
                return;
            }
            let device = Device::default();
            let config = fused_config(false, false, 0.1);
            let fused = config.clone().build();
            let reference = config.with_execution(None).build();
            let original = StreamId::current();
            let allocation = StreamId::allocate();
            let submission = StreamId::allocate();
            for pin in [None, Some(StreamId::allocate())] {
                let (param, grad) = allocation.executes(|| {
                    (
                        tensor(&[1.0, -2.0, 3.0, -4.0], [4], &device),
                        tensor(&[0.25, -0.5, 0.75, -1.0], [4], &device),
                    )
                });
                let expected = reference.step(LEARNING_RATE, param.clone(), grad.clone(), None);
                let mut primitive = param.try_into_primitive::<burn_cubecl::Cube>().unwrap();
                let raw = (&mut primitive as &mut dyn core::any::Any)
                    .downcast_mut::<burn_cubecl::tensor::CubeTensor>()
                    .unwrap();
                assert_eq!(raw.handle.stream, allocation);
                if let Some(pin) = pin {
                    // The client pin does not change the input's allocation stream.
                    unsafe { raw.client.set_stream(pin) };
                }
                let param = Tensor::<1>::from_primitive::<burn_cubecl::Cube>(primitive);
                submission.executes(|| {
                    let actual = one_launch(6, || fused.step(LEARNING_RATE, param, grad, None));
                    let state = actual.1.as_ref().unwrap();
                    for moment in [&state.momentum.moment_1, &state.momentum.moment_2] {
                        let primitive = moment
                            .clone()
                            .try_into_primitive::<burn_cubecl::Cube>()
                            .unwrap();
                        let raw = (&primitive as &dyn core::any::Any)
                            .downcast_ref::<burn_cubecl::tensor::CubeTensor>()
                            .unwrap();
                        assert_eq!(raw.handle.stream, pin.unwrap_or(submission));
                    }
                    assert_step_close(&expected, &actual);
                    assert_eq!(StreamId::current(), submission);
                });
                assert_eq!(StreamId::current(), original);
            }
        }

        #[test]
        fn cow_reuses_each_owned_output_and_preserves_shared_state() {
            let _guard = step_guard();
            if kernel_count(10) == 0 {
                return;
            }
            let device = Device::default();
            let config = fused_config(true, false, 0.1);
            let fused = config.build();
            let reference = config.with_execution(None).build();
            let param = tensor(&values(16, 1.0), [4, 4], &device);
            let grad = tensor(&values(16, 0.1), [4, 4], &device);
            let original = allocation(&param);
            let expected = reference.step(LEARNING_RATE, param.clone(), grad.clone(), None);
            // Reuse assertions require the reference's outstanding reads to have completed.
            device.sync().unwrap();
            let mut actual = one_launch(7, || fused.step(LEARNING_RATE, param, grad.clone(), None));
            assert_step_close(&expected, &actual);
            assert_eq!(
                allocation(&actual.0),
                original,
                "initial parameter should reuse its buffer"
            );

            for shared in [false, true] {
                let ids = output_allocations(&actual.0, actual.1.as_ref().unwrap());
                // Retaining only m1 must not prevent independent param/m2/max reuse.
                let retained = shared.then(|| actual.1.as_ref().unwrap().momentum.moment_1.clone());
                let retained_data = retained.as_ref().map(Tensor::to_data);
                let expected = reference.step(
                    LEARNING_RATE,
                    actual.0.clone(),
                    grad.clone(),
                    actual.1.clone(),
                );
                device.sync().unwrap();
                actual = one_launch(10, || {
                    fused.step(LEARNING_RATE, actual.0, grad.clone(), actual.1)
                });
                assert_step_close(&expected, &actual);
                let next = output_allocations(&actual.0, actual.1.as_ref().unwrap());
                for role in 0..4 {
                    assert_eq!(
                        ids[role] == next[role],
                        !(shared && role == 1),
                        "wrong COW decision for role {role}"
                    );
                }
                if let (Some(retained), Some(data)) = (retained, retained_data) {
                    retained.to_data().assert_eq(&data, true);
                }
            }
        }

        #[test]
        fn cow_refuses_aliases_and_handles_layouts() {
            let _guard = step_guard();
            if kernel_count(10) == 0 {
                return;
            }
            let device = Device::default();
            let config = fused_config(true, false, 0.1);
            let fused = config.build();
            let reference = config.with_execution(None).build();
            for layout in 0..3 {
                let param = match layout {
                    0 => tensor(&values(16, 1.0), [4, 4], &device),
                    1 => tensor(&values(16, 1.0), [4, 4], &device).transpose(),
                    _ => tensor(&values(32, 1.0), [4, 8], &device).narrow(1, 2, 4),
                };
                let grad = if layout != 0 {
                    tensor(&values(16, 0.1), [4, 4], &device)
                } else {
                    param.clone()
                };
                // Some runtimes materialize a narrow into contiguous storage. That fresh,
                // exclusive buffer is reusable; a real strided view remains out-of-place.
                let contiguous = {
                    let primitive = param
                        .clone()
                        .try_into_primitive::<burn_cubecl::Cube>()
                        .unwrap();
                    (&primitive as &dyn core::any::Any)
                        .downcast_ref::<burn_cubecl::tensor::CubeTensor>()
                        .unwrap()
                        .is_contiguous()
                };
                let moments = tensor(&[0.1; 16], [4, 4], &device);
                let state = AdamWState {
                    momentum: AdaptiveMomentumState {
                        time: 2,
                        moment_1: moments.clone(),
                        moment_2: moments.clone(),
                        max_moment_2: Some(moments),
                    },
                };
                let ids = output_allocations(&param, &state);
                let expected = reference.step(
                    LEARNING_RATE,
                    param.clone(),
                    grad.clone(),
                    Some(state.clone()),
                );
                device.sync().unwrap();
                let actual = one_launch(10, || fused.step(LEARNING_RATE, param, grad, Some(state)));
                assert_step_close(&expected, &actual);
                let outputs = output_allocations(&actual.0, actual.1.as_ref().unwrap());
                assert_eq!(outputs[0] == ids[0], layout != 0 && contiguous);
                assert!(
                    outputs[1..].iter().all(|id| !ids.contains(id)),
                    "aliased moments must use fresh outputs"
                );
            }
        }

        #[test]
        fn cow_external_clones_keep_the_old_parameter_and_all_moments() {
            let _guard = step_guard();
            if kernel_count(10) == 0 {
                return;
            }
            let device = Device::default();
            let fused = fused_config(true, false, 0.1).build();
            let param = tensor(&values(16, 1.0), [4, 4], &device);
            let grad = tensor(&values(16, 0.1), [4, 4], &device);
            let (param, state) = fused.step(LEARNING_RATE, param, grad.clone(), None);
            let state = state.unwrap();
            let retained = [
                param.clone(),
                state.momentum.moment_1.clone(),
                state.momentum.moment_2.clone(),
                state.momentum.max_moment_2.as_ref().unwrap().clone(),
            ];
            let before = retained.each_ref().map(Tensor::to_data);
            let ids = output_allocations(&param, &state);
            let (param, state) =
                one_launch(10, || fused.step(LEARNING_RATE, param, grad, Some(state)));
            let next = output_allocations(&param, state.as_ref().unwrap());
            assert!(next.iter().all(|id| !ids.contains(id)));
            for (tensor, data) in retained.into_iter().zip(before) {
                tensor.to_data().assert_eq(&data, true);
            }
        }

        #[test]
        fn cow_keeps_a_preserved_maximum_unchanged() {
            let _guard = step_guard();
            if kernel_count(8) == 0 {
                return;
            }
            let device = Device::default();
            let param = tensor(&values(16, 1.0), [4, 4], &device);
            let grad = tensor(&values(16, 0.1), [4, 4], &device);
            let m2 = tensor(&[0.2; 16], [4, 4], &device);
            let maximum_id = allocation(&m2);
            let state = AdamWState {
                momentum: AdaptiveMomentumState {
                    time: 2,
                    moment_1: tensor(&[0.1; 16], [4, 4], &device),
                    moment_2: m2.clone(),
                    max_moment_2: Some(m2),
                },
            };
            let fused = fused_config(false, false, 0.1).build();
            let (_, state) = one_launch(8, || fused.step(LEARNING_RATE, param, grad, Some(state)));
            let state = state.unwrap();
            assert_ne!(allocation(&state.momentum.moment_2), maximum_id);
            let maximum = state.momentum.max_moment_2.unwrap();
            assert_eq!(allocation(&maximum), maximum_id);
            assert_eq!(maximum.to_data().as_slice::<f32>().unwrap(), &[0.2; 16]);
        }

        #[test]
        fn primitive_rejection_reconstructs_state_for_reference() {
            let _guard = step_guard();
            let device = Device::default();
            let param = tensor(&values(16, 1.0), [4, 4], &device);
            // Zero strides pass the high-level shape check, then the primitive boundary
            // declines after successful conversions and release of the wrapper handles.
            let grad = tensor(&[0.1], [1, 1], &device).expand([4, 4]);
            let state = AdamWState {
                momentum: AdaptiveMomentumState {
                    time: 7,
                    moment_1: tensor(&[0.1; 16], [4, 4], &device),
                    moment_2: tensor(&[0.2; 16], [4, 4], &device),
                    max_moment_2: Some(tensor(&[0.3; 16], [4, 4], &device)),
                },
            };
            let config = fused_config(false, false, 0.1);
            let reference = config.clone().with_execution(None).build();
            let expected = reference.step(
                LEARNING_RATE,
                param.clone(),
                grad.clone(),
                Some(state.clone()),
            );
            let actual = no_launch(|| config.build().step(LEARNING_RATE, param, grad, Some(state)));
            assert_step_close(&expected, &actual);
            assert_eq!(actual.1.unwrap().momentum.time, 8);
        }

        #[test]
        fn cow_capture_replay_matches_one_eager_update() {
            let _guard = step_guard();
            if kernel_count(10) == 0 {
                return;
            }
            let device = Device::default();
            let client = match device.as_dispatch() {
                burn::backend::DispatchDevice::Cube(cube) => {
                    if cube.runtime() == burn_cubecl::cubecl::RuntimeId::Cpu {
                        return;
                    }
                    cube.client()
                }
                #[allow(unreachable_patterns)]
                _ => return,
            };
            let config = fused_config(true, false, 0.1);
            let fused = config.build();
            let reference = config.with_execution(None).build();
            let grad = tensor(&values(16, 0.1), [4, 4], &device);
            let (param, state) = fused.step(
                LEARNING_RATE,
                tensor(&values(16, 1.0), [4, 4], &device),
                grad.clone(),
                None,
            );
            client.graph_prepare().unwrap();
            let (param, state) = fused.step(LEARNING_RATE, param, grad.clone(), state);
            let expected =
                reference.step(LEARNING_RATE, param.clone(), grad.clone(), state.clone());
            device.sync().unwrap();
            let before = output_allocations(&param, state.as_ref().unwrap());
            client.start_capture().unwrap();
            let actual = fused.step(LEARNING_RATE, param, grad.clone(), state);
            let graph = client.stop_capture().unwrap();
            assert_eq!(
                output_allocations(&actual.0, actual.1.as_ref().unwrap()),
                before
            );
            // Every captured input is kept alive by grad or the reused outputs, on this
            // same thread/stream, with no concurrent access until replay completes.
            unsafe {
                graph.replay().unwrap();
            }
            device.sync().unwrap();
            assert_step_close(&expected, &actual);
            drop(graph);
            drop(grad);
        }

        #[test]
        fn vectorized_steps_cover_odd_lengths_and_unaligned_offsets() {
            let _guard = step_guard();
            if kernel_count(10) == 0 {
                return;
            }
            let device = Device::default();
            // WGPU requires storage-buffer binding offsets aligned to its adapter limit.
            // Other runtimes exercise the fused kernel's one-float-offset scalar fallback.
            let offset_stride = match device.as_dispatch() {
                burn::backend::DispatchDevice::Cube(cube)
                    if cube.runtime() == burn_cubecl::cubecl::RuntimeId::Wgpu =>
                {
                    cube.client().properties().memory.alignment.div_ceil(4) as usize
                }
                #[allow(unreachable_patterns)]
                _ => 1,
            };
            let config = fused_config(true, true, 0.1);
            let fused = config.build();
            let reference = config.with_execution(None).build();
            // Adjacent lengths exercise scalar, short-vector and full-vector variants.
            for len in [1, 7, 8, 9, 31, 32, 33, 4095, 4096, 4097] {
                for (param_offset, grad_offset) in [(0, 0), (1, 0), (0, 1)] {
                    let make = |scale, offset| {
                        let offset = offset * offset_stride;
                        let values = values(len + offset, scale);
                        // The reference's generic vector kernels assume aligned bindings.
                        // Give it the same logical values in a normal allocation; only the
                        // fused kernel is responsible for this synthetic unaligned input.
                        let reference = tensor(&values[offset..], [len], &device);
                        let mut primitive = tensor(&values, [len + offset], &device)
                            .try_into_primitive::<burn_cubecl::Cube>()
                            .unwrap();
                        if offset != 0 {
                            let raw = (&mut primitive as &mut dyn core::any::Any)
                                .downcast_mut::<burn_cubecl::tensor::CubeTensor>()
                                .unwrap();
                            raw.handle.offset_start = Some((offset * 4) as u64);
                            raw.meta.remove(0);
                            raw.meta.insert(0, len, 1);
                        }
                        (
                            Tensor::<1>::from_primitive::<burn_cubecl::Cube>(primitive),
                            reference,
                        )
                    };
                    let (param, reference_param) = make(1.0, param_offset);
                    let mut actual = (param, None);
                    let mut expected = (reference_param, None);
                    for _ in 0..3 {
                        let (grad, reference_grad) = make(0.1, grad_offset);
                        expected =
                            reference.step(LEARNING_RATE, expected.0, reference_grad, expected.1);
                        let (launches, _observation) = KernelCounter::start();
                        actual = fused.step(LEARNING_RATE, actual.0, grad, actual.1);
                        assert!(
                            (1..=2).contains(&launches.get()),
                            "expected a vector/scalar update, optionally with a tail"
                        );
                        assert_step_close(&expected, &actual);
                    }
                }
            }
        }

        #[test]
        #[ignore = "process-wide JIT collection: run this test alone with --ignored --test-threads=1"]
        fn vector_width_has_distinct_and_reusable_jit_keys() {
            let _guard = step_guard();
            if kernel_count(6) == 0 {
                return;
            }
            let device = Device::default();
            let client = match device.as_dispatch() {
                burn::backend::DispatchDevice::Cube(cube)
                    if cube.runtime() == burn_cubecl::cubecl::RuntimeId::Cpu =>
                {
                    cube.client()
                }
                #[allow(unreachable_patterns)]
                _ => return,
            };
            let width = client.io_optimized_vector_sizes(4).max().unwrap_or(1);
            if width == 1 {
                return;
            }
            let size = 65536;
            assert_eq!(
                burn_cubecl::cubecl::prelude::CubeDim::new(&client, size / width),
                burn_cubecl::cubecl::prelude::CubeDim::new(&client, size + 1)
            );
            let fused = fused_config(false, false, 0.1).build();
            let run = |len, lr| {
                let param = Tensor::<1>::ones([len], &device);
                let grad = Tensor::<1>::full([len], 0.1, &device);
                device.sync().unwrap();
                let collection = burn_cubecl::cubecl::launched::LaunchedKernels::new();
                let (launches, _observation) = KernelCounter::start();
                let output = fused.step(lr, param, grad, None);
                let count = if len == size { 1 } else { 2 };
                assert_eq!(launches.get(), count);
                let keys = collection.finish();
                device.sync().unwrap();
                assert_eq!(
                    keys.len(),
                    count,
                    "run this test alone so only AdamW is collected"
                );
                drop(output);
                keys
            };
            let vector = run(size, 0.01);
            let with_tail = run(size + 1, 0.01);
            assert!(
                vector.is_subset(&with_tail),
                "the vector body should reuse its JIT key; the scalar tail adds its own"
            );
            assert_eq!(
                vector,
                run(size, 0.02),
                "runtime lr changes must reuse the kernel key"
            );
        }

        fn assert_state_close<const D: usize>(expected: &AdamWState<D>, actual: &AdamWState<D>) {
            assert_eq!(expected.momentum.time, actual.momentum.time);
            for (expected, actual) in [
                (&expected.momentum.moment_1, &actual.momentum.moment_1),
                (&expected.momentum.moment_2, &actual.momentum.moment_2),
            ] {
                expected
                    .to_data()
                    .assert_approx_eq::<f32>(&actual.to_data(), tolerance());
            }
            match (
                &expected.momentum.max_moment_2,
                &actual.momentum.max_moment_2,
            ) {
                (Some(expected), Some(actual)) => {
                    expected
                        .to_data()
                        .assert_approx_eq::<f32>(&actual.to_data(), tolerance());
                }
                (None, None) => {}
                _ => panic!("both states must record or omit the AMSGrad maximum"),
            }
        }

        fn assert_step_close<const D: usize>(
            expected: &(Tensor<D>, Option<AdamWState<D>>),
            actual: &(Tensor<D>, Option<AdamWState<D>>),
        ) {
            expected
                .0
                .to_data()
                .assert_approx_eq::<f32>(&actual.0.to_data(), tolerance());
            assert_state_close(
                expected.1.as_ref().expect("reference state"),
                actual.1.as_ref().expect("fused state"),
            );
        }

        /// Compare values element-wise with per-element rules: NaN is classified (payloads are
        /// not part of the contract), infinities and signed zeros must match bit for bit, and
        /// ordinary values use the numeric tolerance.
        fn assert_special_close(expected: &TensorData, actual: &TensorData) {
            assert_eq!(expected.shape(), actual.shape());
            let expected = expected.as_slice::<f32>().unwrap();
            let actual = actual.as_slice::<f32>().unwrap();
            for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
                if expected.is_nan() || actual.is_nan() {
                    assert!(
                        expected.is_nan() && actual.is_nan(),
                        "element {index}: {expected} != {actual}"
                    );
                    continue;
                }
                let special = expected.is_infinite()
                    || actual.is_infinite()
                    || (*expected == 0.0 && *actual == 0.0);
                if special {
                    assert_eq!(
                        expected.to_bits(),
                        actual.to_bits(),
                        "element {index}: {expected} ({:#x}) != {actual} ({:#x})",
                        expected.to_bits(),
                        actual.to_bits(),
                    );
                } else {
                    let error = (expected - actual).abs();
                    let bound =
                        8.0 * f32::EPSILON * expected.abs().max(actual.abs()) + 4.0 * f32::EPSILON;
                    assert!(
                        error <= bound,
                        "element {index}: {expected} != {actual} (error {error} > {bound})"
                    );
                }
            }
        }

        fn assert_special_state_close<const D: usize>(
            expected: &AdamWState<D>,
            actual: &AdamWState<D>,
        ) {
            assert_eq!(expected.momentum.time, actual.momentum.time);
            assert_special_close(
                &expected.momentum.moment_1.to_data(),
                &actual.momentum.moment_1.to_data(),
            );
            assert_special_close(
                &expected.momentum.moment_2.to_data(),
                &actual.momentum.moment_2.to_data(),
            );
            match (
                &expected.momentum.max_moment_2,
                &actual.momentum.max_moment_2,
            ) {
                (Some(expected), Some(actual)) => {
                    assert_special_close(&expected.to_data(), &actual.to_data());
                }
                (None, None) => {}
                _ => panic!("both states must record or omit the AMSGrad maximum"),
            }
        }

        #[test]
        fn fused_boundary_accepts_initial_and_state() {
            let _guard = step_guard();
            // In a fusion build the production path declines; the raw CI build runs this.
            if burn_cubecl::FUSED {
                return;
            }
            let device = Device::default();
            let convert = |tensor: Tensor<2>| {
                tensor
                    .try_into_primitive::<burn_cubecl::Cube>()
                    .expect("fused test tensors must be raw CubeCL")
            };
            let param = tensor(&values(16, 1.0), [4, 4], &device);
            let grad = tensor(&values(16, 0.1), [4, 4], &device);
            let moment_1 = tensor(&values(16, 0.01), [4, 4], &device);
            let moment_2 = tensor(&values(16, 0.001), [4, 4], &device);
            let scalars = AdamWHostScalars::new(LEARNING_RATE, 0.9, 0.999, 1e-8, 0.5, 2);

            let initial = AdamWInputs::Initial {
                param: convert(param.clone()),
                grad: convert(grad.clone()),
            };
            assert_eq!(
                try_adamw(scalars, false, false, initial).is_ok(),
                kernel_count(6) == 1,
                "the production boundary rejects an initial step"
            );

            let stateful = AdamWInputs::State {
                param: convert(param),
                grad: convert(grad),
                moment_1: convert(moment_1),
                moment_2: convert(moment_2),
                max_moment_2: None,
            };
            assert_eq!(
                try_adamw(scalars, false, false, stateful).is_ok(),
                kernel_count(8) == 1,
                "the production boundary rejects a stateful step"
            );
        }

        #[test]
        fn cautious_decay_matches_reference_through_the_kernel() {
            let _guard = step_guard();
            let device = Device::default();
            let fused = fused_config(false, true, 0.5).build();
            let reference = fused_config(false, true, 0.5).with_execution(None).build();
            let param = tensor(&values(16, 1.0), [4, 4], &device);
            let grad = tensor(&values(16, 0.1), [4, 4], &device);

            let fused_step = one_launch(6, || {
                fused.step(LEARNING_RATE, param.clone(), grad.clone(), None)
            });
            let reference_step = reference.step(LEARNING_RATE, param, grad, None);
            assert_step_close(&reference_step, &fused_step);
        }

        #[test]
        fn cautious_zero_crossing_keeps_separately_rounded_moments() {
            let _guard = step_guard();
            if burn_cubecl::FUSED {
                return;
            }
            let device = Device::default();
            let param = tensor(&[1.0], [1], &device);
            let grad = tensor(&[1.0], [1], &device);
            let state = AdamWState {
                momentum: AdaptiveMomentumState {
                    time: 1,
                    // Rounded old*beta and grad*(1-beta) cancel to +0; contracting
                    // the first product instead gives a negative residual and skips decay.
                    moment_1: tensor(&[f32::from_bits(0xbde38e3d)], [1], &device),
                    moment_2: tensor(&[1.0], [1], &device),
                    max_moment_2: None,
                },
            };
            let config = fused_config(false, true, 0.5);
            let reference = config.clone().with_execution(None).build();
            let expected = reference.step(0.1, param.clone(), grad.clone(), Some(state.clone()));
            let actual = one_launch(8, || config.build().step(0.1, param, grad, Some(state)));
            assert_eq!(
                expected
                    .1
                    .as_ref()
                    .unwrap()
                    .momentum
                    .moment_1
                    .to_data()
                    .as_slice::<f32>()
                    .unwrap(),
                &[0.0]
            );
            assert_step_close(&expected, &actual);
        }

        #[test]
        fn rank_zero_is_rejected() {
            let _guard = step_guard();
            let device = Device::default();
            let param: Tensor<0> =
                Tensor::<1>::from_floats([1.0], &device).reshape::<0, [usize; 0]>([]);
            let grad: Tensor<0> =
                Tensor::<1>::from_floats([0.1], &device).reshape::<0, [usize; 0]>([]);
            let convert = |tensor: Tensor<0>| {
                tensor
                    .try_into_primitive::<burn_cubecl::Cube>()
                    .expect("rank-0 tensor conversion")
            };
            let scalars = AdamWHostScalars::new(LEARNING_RATE, 0.9, 0.999, 1e-8, 0.5, 1);
            let inputs = AdamWInputs::Initial {
                param: convert(param),
                grad: convert(grad),
            };
            assert!(
                try_adamw(scalars, false, false, inputs).is_err(),
                "rank 0 must be rejected by the fused path"
            );
        }

        #[test]
        fn fused_step_reaches_the_kernel() {
            let _guard = step_guard();
            let (launches, _observation) = KernelCounter::start();
            if burn_cubecl::FUSED {
                return;
            }
            let device = Device::default();
            let fused = fused_config(true, false, 0.5).build();
            let reference = fused_config(true, false, 0.5).with_execution(None).build();
            let param = tensor(&values(16, 1.0), [4, 4], &device);
            let grad = tensor(&values(16, 0.1), [4, 4], &device);

            let before = launches.get();
            let (_, state) = fused.step(LEARNING_RATE, param.clone(), grad.clone(), None);
            assert_eq!(
                launches.get() - before,
                kernel_count(7),
                "the initial fused step did not reach the kernel"
            );

            let before = launches.get();
            fused.step(LEARNING_RATE, param.clone(), grad.clone(), state);
            assert_eq!(
                launches.get() - before,
                kernel_count(10),
                "the stateful fused step did not reach the kernel"
            );

            let before = launches.get();
            reference.step(LEARNING_RATE, param, grad, None);
            assert_eq!(
                launches.get() - before,
                0,
                "the reference step must not submit the fused kernel"
            );
        }

        #[test]
        fn direct_step_on_autodiff_tensor_falls_back() {
            let _guard = step_guard();
            let device = Device::default().autodiff();
            let fused = fused_config(false, false, 0.5).build();
            let reference = fused_config(false, false, 0.5).with_execution(None).build();

            let param = Tensor::<2>::ones([4, 4], &device).require_grad();
            let grad = Tensor::<2>::ones([4, 4], &device);
            let fused_step =
                no_launch(|| fused.step(LEARNING_RATE, param.clone(), grad.clone(), None));
            let reference_step = reference.step(LEARNING_RATE, param, grad, None);
            assert_step_close(&reference_step, &fused_step);
            assert_eq!(
                fused_step.0.is_autodiff(),
                reference_step.0.is_autodiff(),
                "a direct step must not drop the autodiff context"
            );
            assert_eq!(
                fused_step.0.is_require_grad(),
                reference_step.0.is_require_grad(),
            );
        }

        #[test]
        fn matches_reference_over_steps() {
            let _guard = step_guard();
            let (launches, _observation) = KernelCounter::start();
            let before = launches.get();
            let device = Device::default();
            for (amsgrad, cautious) in [(false, false), (true, false), (false, true), (true, true)]
            {
                let reference = fused_config(amsgrad, cautious, 0.5)
                    .with_execution(None)
                    .build();
                let fused = fused_config(amsgrad, cautious, 0.5).build();

                let mut reference_param = tensor(&values(64, 1.0), [8, 8], &device);
                let mut fused_param = reference_param.clone();
                let mut reference_state = None;
                let mut fused_state = None;
                for step in 0..5 {
                    let grad = tensor(&values(64, 0.1 * (step + 1) as f32), [8, 8], &device);
                    let reference_step = reference.step(
                        LEARNING_RATE,
                        reference_param,
                        grad.clone(),
                        reference_state,
                    );
                    let fused_step = fused.step(LEARNING_RATE, fused_param, grad, fused_state);
                    assert_step_close(&reference_step, &fused_step);
                    reference_param = reference_step.0;
                    fused_param = fused_step.0;
                    reference_state = reference_step.1;
                    fused_state = fused_step.1;
                }
            }
            assert_eq!(
                launches.get() - before,
                2 * (kernel_count(6)
                    + 4 * kernel_count(8)
                    + kernel_count(7)
                    + 4 * kernel_count(10)),
                "the fused steps did not all reach the kernel"
            );
        }

        #[test]
        fn matches_reference_with_special_values() {
            let _guard = step_guard();
            let (launches, _observation) = KernelCounter::start();
            let device = Device::default();
            let special = [
                -0.0,
                0.0,
                -1e-30,
                1e-30,
                f32::NAN,
                f32::INFINITY,
                f32::NEG_INFINITY,
                1e-40,
            ];
            let m1_values = [1e-30, -1e-30, 1e-30, -1e-30, 0.0, -0.0, 1.0, -1.0];
            let m2_values = [
                1e-12,
                1e-12,
                1e-12,
                1e-12,
                f32::NAN,
                f32::INFINITY,
                1e-12,
                1e-12,
            ];

            for (amsgrad, cautious) in [(false, false), (true, false), (false, true), (true, true)]
            {
                let reference = fused_config(amsgrad, cautious, 0.5)
                    .with_execution(None)
                    .build();
                let fused = fused_config(amsgrad, cautious, 0.5).build();

                let param = tensor(&special, [8], &device);
                let grad = tensor(&[0.0; 8], [8], &device);
                let state = AdamWState {
                    momentum: AdaptiveMomentumState {
                        time: 5,
                        moment_1: tensor(&m1_values, [8], &device),
                        moment_2: tensor(&m2_values, [8], &device),
                        max_moment_2: amsgrad.then(|| tensor(&special, [8], &device)),
                    },
                };

                let reference_step = reference.step(
                    LEARNING_RATE,
                    param.clone(),
                    grad.clone(),
                    Some(state.clone()),
                );
                let before = launches.get();
                let fused_step = fused.step(LEARNING_RATE, param, grad, Some(state));
                assert_eq!(
                    launches.get() - before,
                    kernel_count(if amsgrad { 10 } else { 8 }),
                );

                assert_special_close(&reference_step.0.to_data(), &fused_step.0.to_data());
                assert_special_state_close(
                    reference_step.1.as_ref().unwrap(),
                    fused_step.1.as_ref().unwrap(),
                );
            }
        }

        #[test]
        fn max_state_follows_amsgrad_switch() {
            let _guard = step_guard();
            let device = Device::default();
            let param = tensor(&values(16, 1.0), [4, 4], &device);
            let grad = tensor(&values(16, 0.3), [4, 4], &device);
            let with_max = AdamWState {
                momentum: AdaptiveMomentumState {
                    time: 3,
                    moment_1: tensor(&values(16, 0.01), [4, 4], &device),
                    moment_2: tensor(
                        &values(16, 0.001)
                            .iter()
                            .map(|v| v.abs())
                            .collect::<Vec<_>>(),
                        [4, 4],
                        &device,
                    ),
                    max_moment_2: Some(tensor(
                        &values(16, 0.002)
                            .iter()
                            .map(|v| v.abs())
                            .collect::<Vec<_>>(),
                        [4, 4],
                        &device,
                    )),
                },
            };

            // AMSGrad state without a maximum: the new second moment initializes it.
            let without_max = AdamWState {
                momentum: AdaptiveMomentumState {
                    max_moment_2: None,
                    ..with_max.momentum.clone()
                },
            };
            for state in [with_max.clone(), without_max.clone()] {
                for amsgrad in [false, true] {
                    let reference = fused_config(amsgrad, false, 0.1)
                        .with_execution(None)
                        .build();
                    let fused = fused_config(amsgrad, false, 0.1).build();
                    let reference_step = reference.step(
                        LEARNING_RATE,
                        param.clone(),
                        grad.clone(),
                        Some(state.clone()),
                    );
                    let fused_step = one_launch(
                        if amsgrad {
                            if state.momentum.max_moment_2.is_some() {
                                10
                            } else {
                                9
                            }
                        } else {
                            8
                        },
                        || {
                            fused.step(
                                LEARNING_RATE,
                                param.clone(),
                                grad.clone(),
                                Some(state.clone()),
                            )
                        },
                    );
                    assert_step_close(&reference_step, &fused_step);
                }
            }

            // An AMSGrad maximum recorded before a non-AMSGrad step survives it untouched.
            let fused = fused_config(false, false, 0.1).build();
            let (_, state) =
                one_launch(8, || fused.step(LEARNING_RATE, param, grad, Some(with_max)));
            assert!(state.unwrap().momentum.max_moment_2.is_some());
        }

        #[test]
        fn matches_reference_for_ranks_and_strided_layouts() {
            let _guard = step_guard();
            let device = Device::default();
            let reference = fused_config(true, false, 0.1).with_execution(None).build();
            let fused = fused_config(true, false, 0.1).build();

            macro_rules! check {
                ($param:expr, $grad:expr) => {{
                    let reference_step =
                        reference.step(LEARNING_RATE, $param.clone(), $grad.clone(), None);
                    let fused_step =
                        one_launch(7, || fused.step(LEARNING_RATE, $param, $grad, None));
                    assert_step_close(&reference_step, &fused_step);
                }};
            }

            check!(
                tensor(&values(8, 1.0), [8], &device),
                tensor(&values(8, 0.1), [8], &device)
            );
            check!(
                tensor(&values(16, 1.0), [4, 4], &device),
                tensor(&values(16, 0.1), [4, 4], &device)
            );
            check!(
                tensor(&values(24, 1.0), [2, 3, 4], &device),
                tensor(&values(24, 0.1), [2, 3, 4], &device)
            );

            // A transposed parameter is read through its strides, never written.
            let base = tensor::<4>(&values(120, 1.0), [2, 3, 4, 5], &device);
            let param = base.swap_dims(0, 3);
            let grad = tensor(&values(120, 0.1), [5, 3, 4, 2], &device);
            check!(param, grad);

            // A narrowed parameter is non-contiguous but non-overlapping.
            let base = tensor::<4>(&values(240, 1.0), [2, 3, 4, 10], &device);
            let param = base.narrow(3, 2, 5);
            let grad = tensor(&values(120, 0.1), [2, 3, 4, 5], &device);
            check!(param, grad);
        }

        #[test]
        fn matches_reference_over_fifty_steps() {
            let _guard = step_guard();
            let device = Device::default();
            let reference = fused_config(true, false, 0.5).with_execution(None).build();
            let fused = fused_config(true, false, 0.5).build();

            let mut reference_param = tensor(&values(16, 1.0), [4, 4], &device);
            let mut fused_param = reference_param.clone();
            let mut reference_state = None;
            let mut fused_state = None;
            let (launches, _observation) = KernelCounter::start();
            let before = launches.get();
            for step in 0..50 {
                let grad = tensor(&values(16, 0.05 * (step + 1) as f32), [4, 4], &device);
                let reference_step = reference.step(
                    LEARNING_RATE,
                    reference_param,
                    grad.clone(),
                    reference_state,
                );
                let fused_step = fused.step(LEARNING_RATE, fused_param, grad, fused_state);
                assert_step_close(&reference_step, &fused_step);
                reference_param = reference_step.0;
                fused_param = fused_step.0;
                reference_state = reference_step.1;
                fused_state = fused_step.1;
            }
            assert_eq!(
                launches.get() - before,
                kernel_count(7) + 49 * kernel_count(10),
                "the fused steps did not all reach the kernel"
            );
        }

        #[test]
        fn intermediate_update_overflow_matches_reference() {
            let _guard = step_guard();
            let device = Device::default();
            let config = fused_config(false, false, 0.0)
                .with_beta_1(0.5)
                .with_beta_2(0.0)
                .with_epsilon(1.0);
            let reference = config.clone().with_execution(None).build();
            let fused = config.build();
            let param = tensor(&[3e38, -3e38], [2], &device);
            let grad = tensor(&[6.0, -6.0], [2], &device);
            // Reach m1 = +/-3, m2 = 36 through a normal first step.
            let expected = reference.step(0.0, param.clone(), grad.clone(), None);
            let actual = one_launch(6, || fused.step(0.0, param, grad, None));
            let grad = tensor(&[0.0, 0.0], [2], &device);
            // delta = +/-2: the separately rounded product with lr overflows.
            let expected = reference.step(2e38, expected.0, grad.clone(), expected.1);
            let actual = one_launch(8, || fused.step(2e38, actual.0, grad, actual.1));
            let expected_data = expected.0.to_data();
            // Fusion contracts the reference's own tensor chain, so the raw reference's
            // intermediate overflow is only observable without it.
            if !burn_cubecl::FUSED {
                assert_eq!(
                    expected_data.as_slice::<f32>().unwrap(),
                    &[f32::NEG_INFINITY, f32::INFINITY]
                );
            }
            assert_special_close(&expected_data, &actual.0.to_data());
            assert_special_state_close(expected.1.as_ref().unwrap(), actual.1.as_ref().unwrap());
        }

        #[test]
        fn no_decay_leaves_infinite_parameters_alone() {
            let _guard = step_guard();
            let device = Device::default();
            let fused = fused_config(false, false, 0.0).build();
            let reference = fused_config(false, false, 0.0).with_execution(None).build();

            let param = tensor(&[f32::INFINITY, 1.0, -2.0, -0.0], [4], &device);
            let grad = tensor(&[0.5, -0.5, 0.25, 0.75], [4], &device);
            let fused_step = one_launch(6, || {
                fused.step(LEARNING_RATE, param.clone(), grad.clone(), None)
            });
            let reference_step = reference.step(LEARNING_RATE, param, grad, None);
            assert_special_close(&reference_step.0.to_data(), &fused_step.0.to_data());
            assert_special_state_close(
                reference_step.1.as_ref().unwrap(),
                fused_step.1.as_ref().unwrap(),
            );
        }

        #[test]
        fn aliased_inputs_are_only_read() {
            let _guard = step_guard();
            let device = Device::default();
            let fused = fused_config(true, false, 0.1).build();
            let reference = fused_config(true, false, 0.1).with_execution(None).build();

            // The parameter and gradient share one buffer, and so do both moments and the
            // AMSGrad maximum. Fresh outputs mean every read still sees the original values.
            let value = tensor(&values(16, 1.0), [4, 4], &device);
            let moments = tensor(&values(16, 0.1), [4, 4], &device);
            let state = || AdamWState {
                momentum: AdaptiveMomentumState {
                    time: 2,
                    moment_1: moments.clone(),
                    moment_2: moments.clone(),
                    max_moment_2: Some(moments.clone()),
                },
            };

            let fused_step = one_launch(10, || {
                fused.step(LEARNING_RATE, value.clone(), value.clone(), Some(state()))
            });
            let expected = reference.step(
                LEARNING_RATE,
                tensor(&values(16, 1.0), [4, 4], &device),
                tensor(&values(16, 1.0), [4, 4], &device),
                Some(state()),
            );
            assert_step_close(&expected, &fused_step);
        }

        #[test]
        fn unsupported_dtypes_fall_back() {
            let _guard = step_guard();
            let device = Device::default();
            let fused = fused_config(false, false, 0.1).build();
            let reference = fused_config(false, false, 0.1).with_execution(None).build();

            for dtype in [DType::F64, DType::F16, DType::BF16] {
                if !device.supports_dtype(dtype) {
                    continue;
                }
                let param = tensor::<2>(&values(16, 1.0), [4, 4], &device).cast(dtype);
                let grad = tensor::<2>(&values(16, 0.1), [4, 4], &device).cast(dtype);
                let fused_step =
                    no_launch(|| fused.step(LEARNING_RATE, param.clone(), grad.clone(), None));
                let reference_step = reference.step(LEARNING_RATE, param, grad, None);
                assert_step_close(&reference_step, &fused_step);
            }
        }

        #[test]
        fn half_precision_keeps_the_reference_evaluation_boundaries() {
            let _guard = step_guard();
            let device = Device::default();
            if !device.supports_dtype(DType::F16) {
                return;
            }
            let config = fused_config(false, false, 0.0);
            let fused = config.build();
            let reference = config.with_execution(None).build();
            let make = || {
                (
                    tensor::<1>(&[1.0], [1], &device).cast(DType::F16),
                    tensor::<1>(&[300.0], [1], &device).cast(DType::F16),
                )
            };
            let (param, grad) = make();
            let expected = reference.step(LEARNING_RATE, param, grad, None);
            let (param, grad) = make();
            let actual = no_launch(|| fused.step(LEARNING_RATE, param, grad, None));
            let expected_state = &expected.1.unwrap().momentum;
            let actual_state = &actual.1.unwrap().momentum;
            assert_eq!(expected_state.time, actual_state.time);
            assert!(expected_state.max_moment_2.is_none());
            assert!(actual_state.max_moment_2.is_none());
            for (expected, actual) in [
                (&expected.0, &actual.0),
                (&expected_state.moment_1, &actual_state.moment_1),
                (&expected_state.moment_2, &actual_state.moment_2),
            ] {
                assert_eq!(actual.dtype(), DType::F16);
                assert_special_close(
                    &expected.clone().cast(DType::F32).to_data(),
                    &actual.clone().cast(DType::F32).to_data(),
                );
            }
            // Eager F16 materializes 300 * 300 as infinity before scaling. A fusion graph can
            // retain a wider intermediate, so its contract is the independently built reference.
            if !burn_cubecl::FUSED {
                let moment_2 = actual_state.moment_2.clone().cast(DType::F32).to_data();
                assert!(moment_2.as_slice::<f32>().unwrap()[0].is_infinite());
            }
        }

        #[test]
        fn empty_parameters_fall_back() {
            let _guard = step_guard();
            let device = Device::default();
            let fused = fused_config(false, false, 0.1).build();
            let reference = fused_config(false, false, 0.1).with_execution(None).build();

            let param = tensor::<2>(&[], [0, 4], &device);
            let grad = tensor::<2>(&[], [0, 4], &device);
            let fused_step =
                no_launch(|| fused.step(LEARNING_RATE, param.clone(), grad.clone(), None));
            let reference_step = reference.step(LEARNING_RATE, param, grad, None);
            assert_step_close(&reference_step, &fused_step);
        }

        #[test]
        fn module_step_matches_reference_for_mixed_rank() {
            let _guard = step_guard();
            let (launches, _observation) = KernelCounter::start();
            for amsgrad in [false, true] {
                let device = Device::default().autodiff();
                let linear = given_linear_layer(
                    TensorData::from([
                        [-0.3206, 0.1374, 0.4043],
                        [0.0777, -0.0185, -0.3667],
                        [-0.0190, 0.0346, -0.2962],
                    ]),
                    TensorData::from([-0.3905, 0.0884, -0.0970]),
                    &device,
                );
                let x = Tensor::<2>::from_floats(
                    [[0.6294, 0.0940, 0.8176], [0.7152, 0.9559, 0.7893]],
                    &device,
                )
                .require_grad();

                let mut reference = fused_config(amsgrad, false, 0.5)
                    .with_execution(None)
                    .init();
                let mut fused = fused_config(amsgrad, false, 0.5).init();

                let mut reference_model = linear.clone();
                let mut fused_model = linear;
                for step in 0..3 {
                    let grads = reference_model.forward(x.clone()).backward();
                    let grads = GradientsParams::from_grads(grads, &reference_model);
                    reference_model = reference.step(LEARNING_RATE, reference_model, grads);

                    let grads = fused_model.forward(x.clone()).backward();
                    let grads = GradientsParams::from_grads(grads, &fused_model);
                    let before = launches.get();
                    fused_model = fused.step(LEARNING_RATE, fused_model, grads);
                    if step == 0 && !burn_cubecl::FUSED {
                        assert_eq!(
                            launches.get() - before,
                            2 * kernel_count(if amsgrad { 7 } else { 6 }),
                            "the module path did not reach the fused kernel"
                        );
                    }
                }

                reference_model
                    .weight
                    .val()
                    .to_data()
                    .assert_approx_eq::<f32>(&fused_model.weight.val().to_data(), tolerance());
                reference_model
                    .bias
                    .as_ref()
                    .unwrap()
                    .val()
                    .to_data()
                    .assert_approx_eq::<f32>(
                        &fused_model.bias.as_ref().unwrap().val().to_data(),
                        tolerance(),
                    );
            }
        }

        #[test]
        fn module_step_applies_grad_clipping_before_the_fused_update() {
            let _guard = step_guard();
            let device = Device::default().autodiff();
            let linear = given_linear_layer(
                TensorData::from([
                    [-0.3206, 0.1374, 0.4043],
                    [0.0777, -0.0185, -0.3667],
                    [-0.0190, 0.0346, -0.2962],
                ]),
                TensorData::from([-0.3905, 0.0884, -0.0970]),
                &device,
            );
            let x = Tensor::<2>::from_floats(
                [[0.6294, 0.0940, 0.8176], [0.7152, 0.9559, 0.7893]],
                &device,
            )
            .require_grad();

            for clipping in [
                GradientClippingConfig::Value(0.01),
                GradientClippingConfig::Norm(0.1),
            ] {
                let mut reference = fused_config(true, false, 0.5)
                    .with_execution(None)
                    .with_grad_clipping(Some(clipping.clone()))
                    .init();
                let mut fused = fused_config(true, false, 0.5)
                    .with_grad_clipping(Some(clipping))
                    .init();

                let mut reference_model = linear.clone();
                let mut fused_model = linear.clone();
                for _ in 0..2 {
                    let grads = reference_model.forward(x.clone()).backward();
                    let grads = GradientsParams::from_grads(grads, &reference_model);
                    reference_model = reference.step(LEARNING_RATE, reference_model, grads);

                    let grads = fused_model.forward(x.clone()).backward();
                    let grads = GradientsParams::from_grads(grads, &fused_model);
                    fused_model = fused.step(LEARNING_RATE, fused_model, grads);
                }

                reference_model
                    .weight
                    .val()
                    .to_data()
                    .assert_approx_eq::<f32>(&fused_model.weight.val().to_data(), tolerance());
                reference_model
                    .bias
                    .as_ref()
                    .unwrap()
                    .val()
                    .to_data()
                    .assert_approx_eq::<f32>(
                        &fused_model.bias.as_ref().unwrap().val().to_data(),
                        tolerance(),
                    );
            }
        }

        #[test]
        fn checkpoint_resumes_between_execution_modes() {
            let _guard = step_guard();
            let device = Device::default().autodiff();
            let linear = LinearConfig::new(6, 6).init(&device);

            for amsgrad in [false, true] {
                let make = |fused: bool| {
                    fused_config(amsgrad, false, 0.5)
                        .with_execution(fused.then_some(AdamWExecutionConfig::Fused))
                        .init()
                };
                // Reference checkpoint continued by Fused, and the other way around.
                assert_optimizer_resume_between(
                    || make(false),
                    || make(true),
                    linear.clone(),
                    LEARNING_RATE,
                );
                assert_optimizer_resume_between(
                    || make(true),
                    || make(false),
                    linear.clone(),
                    LEARNING_RATE,
                );
            }
        }
    }

    fn given_linear_layer(weight: TensorData, bias: TensorData, device: &Device) -> Linear {
        Linear {
            weight: Param::from_data(weight, device),
            bias: Some(Param::from_data(bias, device)),
        }
    }
}
