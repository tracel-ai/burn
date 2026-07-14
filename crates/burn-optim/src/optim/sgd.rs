use burn_core as burn;

use super::Optimizer;
use super::decay::{WeightDecay, WeightDecayConfig};
use super::module_optimizer::ModuleOptimizer;
use super::momentum::{Momentum, MomentumConfig, MomentumState};
use crate::LearningRate;
use crate::RecordState;
use crate::grad_clipping::GradientClippingConfig;
use burn::config::Config;
use burn::tensor::Device;
use burn::tensor::Tensor;

/// Configuration to create the [Sgd](Sgd) optimizer.
#[derive(Config, Debug)]
pub struct SgdConfig {
    /// [Weight decay](WeightDecayConfig) config.
    weight_decay: Option<WeightDecayConfig>,
    /// [Momentum](MomentumConfig) config.
    momentum: Option<MomentumConfig>,
    /// [Gradient Clipping](GradientClippingConfig) config.
    gradient_clipping: Option<GradientClippingConfig>,
}

/// Optimizer that implements stochastic gradient descent with momentum.
///
/// The optimizer can be configured with [SgdConfig](SgdConfig).
#[derive(Clone)]
pub struct Sgd {
    momentum: Option<Momentum>,
    weight_decay: Option<WeightDecay>,
}

/// State of [Sgd](Sgd).
#[derive(RecordState, Clone, new)]
pub struct SgdState<const D: usize> {
    /// The current state of the momentum (if any).
    pub momentum: Option<MomentumState<D>>,
}

impl SgdConfig {
    /// Build a [`Sgd`] from the config.
    pub(crate) fn build(&self) -> Sgd {
        Sgd {
            momentum: self.momentum.as_ref().map(Momentum::new),
            weight_decay: self.weight_decay.as_ref().map(WeightDecay::new),
        }
    }

    /// Initializes the SGD optimizer from the configuration.
    pub fn init(&self) -> ModuleOptimizer {
        let mut optim = ModuleOptimizer::from(self.build());
        if let Some(config) = &self.gradient_clipping {
            optim = optim.with_grad_clipping(config.init());
        }
        optim
    }
}

impl Optimizer for Sgd {
    type State<const D: usize> = SgdState<D>;

    fn step<const D: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<D>,
        mut grad: Tensor<D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<D>, Option<Self::State<D>>) {
        let mut state_momentum = None;

        if let Some(state) = state {
            state_momentum = state.momentum;
        }

        if let Some(weight_decay) = &self.weight_decay {
            grad = weight_decay.transform(grad, tensor.clone());
        }

        if let Some(momentum) = &self.momentum {
            let (grad_out, state) = momentum.transform(grad, state_momentum);
            state_momentum = Some(state);
            grad = grad_out;
        }

        let state = SgdState::new(state_momentum);
        let delta = grad.mul_scalar(lr);

        (tensor - delta, Some(state))
    }

    fn to_device<const D: usize>(mut state: Self::State<D>, device: &Device) -> Self::State<D> {
        state.momentum = state.momentum.map(|state| state.to_device(device));
        state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{grad_clipping::GradientClipping, optim::GradientsParams};
    use burn::tensor::{Distribution, Shape};
    use burn_nn::{Linear, LinearConfig};

    const LEARNING_RATE: LearningRate = 0.02;

    #[test]
    fn with_updated_params_should_have_state() {
        let device = Device::default().autodiff();
        let layer = layer(&device);
        let mut optim = sgd_with_all();
        let loss = layer.forward(random_tensor(&device));
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &layer);
        let _layer = optim.step(LEARNING_RATE.into(), layer, grads);

        let record = optim.to_record();

        assert!(!record.is_empty());
    }

    #[test]
    fn without_updated_params_should_not_have_state() {
        let optim = sgd_with_all();
        let record = optim.to_record();
        assert!(record.is_empty());
    }

    #[test]
    fn can_attach_gradient_clipping() {
        let optim = sgd_with_all().with_grad_clipping(GradientClipping::Value(0.5));
        assert!(optim.has_gradient_clipping());
    }

    #[test]
    fn should_load_state() {
        let device = Device::default().autodiff();
        let layer = layer(&device);
        let mut optim = sgd_with_all();
        let loss = layer.forward(random_tensor(&device));
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &layer);
        let _layer = optim.step(LEARNING_RATE.into(), layer, grads);

        let record = optim.to_record();
        let bytes = optim.into_bytes().unwrap();
        let optim_new = sgd_with_all();
        let record_new = optim_new.to_record();
        let optim_new = optim_new.from_bytes(bytes).unwrap();
        let state_restored = optim_new.to_record();

        assert_ne!(record.len(), record_new.len());
        assert_eq!(record.len(), state_restored.len());
    }

    #[test]
    fn lora_finetune_trains_adapter_and_freezes_base() {
        use burn::module::{LoraConfig, Module};
        use burn::tensor::Tolerance;

        let device = Device::default().autodiff();
        let linear = LinearConfig::new(8, 8).init(&device);

        // Snapshot the base weight before applying LoRA / training.
        let base_before = linear.weight.val();

        // Apply LoRA transparently — no change to `Linear` or its forward code.
        let mut model = linear.apply_lora(LoraConfig::new(4, 8.0));
        assert!(model.weight.adapter().is_some());
        let b_before = model.weight.adapter().unwrap().b.val();

        let x = Tensor::<2>::random(Shape::new([16, 8]), Distribution::Default, &device);
        let target = Tensor::<2>::random(Shape::new([16, 8]), Distribution::Default, &device);

        let mut optim = SgdConfig::new().init();

        let mut first_loss = None;
        let mut last_loss = 0.0f32;
        for _ in 0..30 {
            let output = model.forward(x.clone());
            let loss = (output - target.clone()).powf_scalar(2.0).mean();
            last_loss = loss.clone().into_scalar::<f32>();
            first_loss.get_or_insert(last_loss);

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(0.5.into(), model, grads);
        }

        // Training reduced the loss...
        assert!(
            last_loss < first_loss.unwrap(),
            "expected loss to decrease ({} -> {})",
            first_loss.unwrap(),
            last_loss
        );

        // ...the adapter is still attached and its B factor moved away from the zero init...
        assert!(model.weight.adapter().is_some());
        let b_after = model.weight.adapter().unwrap().b.val();
        let b_change = (b_after - b_before).abs().sum().into_scalar::<f32>();
        assert!(
            b_change > 0.0,
            "adapter factor B should be updated by training"
        );

        // ...while the frozen base weight is left untouched.
        model
            .weight
            .base()
            .into_data()
            .assert_approx_eq::<f32>(&base_before.into_data(), Tolerance::default());
    }

    fn random_tensor(device: &Device) -> Tensor<2> {
        Tensor::<2>::random(Shape::new([2, 20]), Distribution::Default, device)
    }

    fn layer(device: &Device) -> Linear {
        LinearConfig::new(20, 20).init(device)
    }

    fn sgd_with_all() -> ModuleOptimizer {
        SgdConfig {
            weight_decay: Some(WeightDecayConfig { penalty: 0.05 }),
            momentum: Some(MomentumConfig {
                momentum: 0.9,
                dampening: 0.1,
                nesterov: true,
            }),
            gradient_clipping: None,
        }
        .init()
    }
}
