use burn_core as burn;

use crate::{LearningRate, RecordState, grad_clipping::GradientClippingConfig};
use burn::config::Config;
use burn::tensor::{Device, Tensor};

use super::{Optimizer, module_optimizer::ModuleOptimizer};

/// Configuration for the [`Lion`] optimizer.
#[derive(Config, Debug)]
pub struct LionConfig {
    /// Interpolation factor used to compute the update direction.
    #[config(default = 0.9)]
    beta_1: f32,
    /// Decay factor for the momentum state.
    #[config(default = 0.99)]
    beta_2: f32,
    /// Decoupled weight decay factor.
    #[config(default = 0.0)]
    weight_decay: f32,
    /// [Gradient clipping](GradientClippingConfig) configuration.
    grad_clipping: Option<GradientClippingConfig>,
}

/// Lion (EvoLved Sign Momentum) optimizer.
///
/// Lion keeps a single momentum tensor and updates parameters with the sign of an
/// interpolation between that momentum and the current gradient. Weight decay is
/// decoupled from the gradient, as in AdamW.
///
/// Lion commonly needs a learning rate smaller than AdamW's and a larger decoupled
/// weight decay. These hyperparameters should be tuned together.
///
/// See [Symbolic Discovery of Optimization Algorithms](https://arxiv.org/abs/2302.06675).
#[derive(Clone)]
pub struct Lion {
    beta_1: f32,
    beta_2: f32,
    weight_decay: f32,
}

/// State of the [`Lion`] optimizer.
#[derive(RecordState, Clone, new)]
pub struct LionState<const D: usize> {
    /// Exponential moving average of the gradients.
    pub momentum: Tensor<D>,
}

impl LionConfig {
    /// Build a [`Lion`] from the config.
    ///
    /// The bare optimizer, which
    /// [`ModuleOptimizer::with_group`](crate::ModuleOptimizer::with_group) takes to
    /// optimize one parameter group. [`init`](Self::init) is the whole-module
    /// counterpart, and the only one that applies the configured gradient clipping.
    pub fn build(&self) -> Lion {
        Lion {
            beta_1: self.beta_1,
            beta_2: self.beta_2,
            weight_decay: self.weight_decay,
        }
    }

    /// Initialize a Lion optimizer for a module.
    pub fn init(&self) -> ModuleOptimizer {
        let mut optim = ModuleOptimizer::from(self.build());
        if let Some(config) = &self.grad_clipping {
            optim = optim.with_grad_clipping(config.init());
        }
        optim
    }
}

impl Optimizer for Lion {
    type State<const D: usize> = LionState<D>;

    fn step<const D: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<D>,
        grad: Tensor<D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<D>, Option<Self::State<D>>) {
        let (update, momentum) = match state {
            Some(state) => {
                let update = state
                    .momentum
                    .clone()
                    .mul_scalar(self.beta_1)
                    .add(grad.clone().mul_scalar(1.0 - self.beta_1))
                    .sign();
                let momentum = state
                    .momentum
                    .mul_scalar(self.beta_2)
                    .add(grad.mul_scalar(1.0 - self.beta_2));

                (update, momentum)
            }
            None => {
                let update = grad.clone().mul_scalar(1.0 - self.beta_1).sign();
                let momentum = grad.mul_scalar(1.0 - self.beta_2);

                (update, momentum)
            }
        };

        let decay = 1.0 - lr * self.weight_decay as f64;
        let tensor = if decay == 1.0 {
            tensor
        } else {
            tensor.mul_scalar(decay)
        };
        let tensor = tensor - update.mul_scalar(lr);

        (tensor, Some(LionState::new(momentum)))
    }

    fn to_device<const D: usize>(mut state: Self::State<D>, device: &Device) -> Self::State<D> {
        state.momentum = state.momentum.to_device(device);
        state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optim::test_utils::assert_optimizer_resume;
    use crate::{AdamWConfig, GradientsParams, ModuleOptimizer};
    use burn::module::Param;
    use burn::tensor::{TensorData, Tolerance};
    use burn_nn::loss::{MseLoss, Reduction};
    use burn_nn::{Linear, LinearConfig};

    #[test]
    fn test_lion_two_steps_with_weight_decay() {
        let device = Device::default();
        let optimizer = LionConfig::new()
            .with_beta_1(0.9)
            .with_beta_2(0.99)
            .with_weight_decay(0.1)
            .build();
        let tensor = Tensor::<1>::from_floats([1.0, -2.0, 3.0], &device);
        let grad = Tensor::<1>::from_floats([0.5, -0.25, 0.0], &device);

        let (tensor, state) = optimizer.step(0.1, tensor, grad, None);
        tensor.clone().into_data().assert_approx_eq::<f32>(
            &TensorData::from([0.89, -1.88, 2.97]),
            Tolerance::absolute(1e-6),
        );
        let state = state.unwrap();
        state.momentum.clone().into_data().assert_approx_eq::<f32>(
            &TensorData::from([0.005, -0.0025, 0.0]),
            Tolerance::absolute(1e-6),
        );

        let grad = Tensor::<1>::from_floats([-0.1, 0.5, -2.0], &device);
        let (tensor, state) = optimizer.step(0.1, tensor, grad, Some(state));
        tensor.into_data().assert_approx_eq::<f32>(
            &TensorData::from([0.9811, -1.9612, 3.0403]),
            Tolerance::absolute(1e-6),
        );
        state.unwrap().momentum.into_data().assert_approx_eq::<f32>(
            &TensorData::from([0.00395, 0.002525, -0.02]),
            Tolerance::absolute(1e-6),
        );
    }

    #[test]
    fn test_lion_optimizer_save_load_state() {
        let device = Device::default().autodiff();
        let linear = LinearConfig::new(4, 3).init(&device);
        assert_optimizer_resume(|| LionConfig::new().init(), linear, 1e-4);
    }

    // A resource-light analogue of the paper's training experiments. It is intentionally
    // deterministic and only verifies that Lion works end-to-end with the paper's guidance of
    // using a lower learning rate than AdamW; it is not an accuracy reproduction of ImageNet.
    #[test]
    fn test_paper_smoke_lion_trains_tiny_regression() {
        let device = Device::default().autodiff();
        let lion_model = tiny_linear_model(&device);
        let adamw_model = tiny_linear_model(&device);

        let (lion_initial, lion_final) =
            train_tiny_regression(lion_model, LionConfig::new().init(), 0.01, &device);
        let (adamw_initial, adamw_final) = train_tiny_regression(
            adamw_model,
            AdamWConfig::new().with_weight_decay(0.0).init(),
            0.05,
            &device,
        );

        println!(
            "tiny regression: Lion {lion_initial:.6e} -> {lion_final:.6e}; \
             AdamW {adamw_initial:.6e} -> {adamw_final:.6e}"
        );
        assert!(
            lion_final < lion_initial * 1e-3,
            "Lion loss did not decrease enough: {lion_initial} -> {lion_final}"
        );
        assert!(adamw_final < adamw_initial * 1e-3);
    }

    // The paper reports that Lion halves the additional optimizer-state memory because it stores
    // one moment instead of AdamW's two. Burnpack size is a stable, backend-independent proxy for
    // the actual tensor payload held by ModuleOptimizer.
    #[test]
    fn test_paper_smoke_lion_state_is_smaller_than_adamw() {
        let device = Device::default().autodiff();
        let model = LinearConfig::new(64, 64).with_bias(false).init(&device);
        let input = Tensor::<2>::ones([2, 64], &device);

        let mut lion = LionConfig::new().init();
        let grads = GradientsParams::from_grads(
            model.clone().forward(input.clone()).sum().backward(),
            &model,
        );
        let _model = lion.step(1e-4, model, grads);
        let lion_state_bytes = lion.into_bytes().unwrap().len();

        let model = LinearConfig::new(64, 64).with_bias(false).init(&device);
        let mut adamw = AdamWConfig::new().with_weight_decay(0.0).init();
        let grads =
            GradientsParams::from_grads(model.clone().forward(input).sum().backward(), &model);
        let _model = adamw.step(1e-3, model, grads);
        let adamw_state_bytes = adamw.into_bytes().unwrap().len();

        println!(
            "serialized optimizer state: Lion {lion_state_bytes} bytes; \
             AdamW {adamw_state_bytes} bytes"
        );
        assert!(
            lion_state_bytes < adamw_state_bytes,
            "expected Lion state ({lion_state_bytes} bytes) to be smaller than AdamW state \
             ({adamw_state_bytes} bytes)"
        );
    }

    fn tiny_linear_model(device: &Device) -> Linear {
        Linear {
            weight: Param::from_data(TensorData::from([[0.0], [0.0]]), device),
            bias: None,
        }
    }

    fn train_tiny_regression(
        mut model: Linear,
        mut optimizer: ModuleOptimizer,
        peak_lr: LearningRate,
        device: &Device,
    ) -> (f32, f32) {
        let input =
            Tensor::<2>::from_floats([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]], device);
        let target = Tensor::<2>::from_floats([[0.5], [-0.75], [-0.25], [-1.25]], device);
        let loss = MseLoss::new();
        let initial = loss
            .forward(
                model.forward(input.clone()),
                target.clone(),
                Reduction::Mean,
            )
            .into_scalar::<f32>();
        let steps = 600;

        for step in 0..steps {
            let output = model.forward(input.clone());
            let value = loss.forward(output, target.clone(), Reduction::Mean);
            let grads = GradientsParams::from_grads(value.backward(), &model);
            let lr = peak_lr * (1.0 - step as f64 / steps as f64);
            model = optimizer.step(lr, model, grads);
        }

        let final_loss = loss
            .forward(model.forward(input), target, Reduction::Mean)
            .into_scalar::<f32>();
        (initial, final_loss)
    }
}
