use burn_core::tensor::{Device, FloatDType, Tensor};

use crate::HostLr;

/// The learning rate of an [optimizer step](crate::Optimizer::step).
///
/// It knows how to [apply](LearningRate::apply) itself to an update, wherever it lives:
///
/// - A [host](LearningRate::Host) value is passed to the kernels as a scalar, the fastest option
///   for eager training.
/// - A [device](LearningRate::Device) value is a `[1]` tensor read by the kernels. Everything the
///   step derives from it, such as a weight decay factor, is computed on the device too.
///
/// # Graph capture
///
/// A captured optimizer step (see `burn::tensor::capture`) needs a device learning rate. A replay
/// runs the recorded kernels only, not the closure: the step's host code, including any
/// learning rate computed or passed as a host value, ran once while recording and is frozen in
/// the graph. With a host learning rate, every replay trains at the capture-time learning rate,
/// silently ignoring the schedule.
///
/// With a device learning rate, the scheduler keeps running on the host, outside the captured
/// closure, and its value is written into the tensor before each replay, the same way the other
/// inputs of the graph are refreshed:
///
/// ```rust,ignore
/// let lr = Tensor::<1>::from_floats([scheduler.step()], &device);
/// let mut graph = capture(&device, || {
///     // Runs while recording only.
///     model = optim.step(lr.clone(), model, grads);
/// });
/// for _ in 0..steps {
///     let value = scheduler.step(); // Host, every step.
///     // Write `value` into `lr`'s buffer, then:
///     unsafe { graph.replay() };
/// }
/// ```
///
/// On a host value, the arithmetic below is plain `f64` arithmetic, so eager training with a host
/// learning rate is unaffected.
#[derive(Clone, Debug)]
#[allow(clippy::large_enum_variant)] // Built once per step, the tensor is only a handle.
pub enum LearningRate {
    /// A learning rate known on the host.
    Host(HostLr),
    /// A learning rate held in a `[1]` device tensor.
    Device(Tensor<1>),
}

impl From<HostLr> for LearningRate {
    fn from(lr: HostLr) -> Self {
        Self::Host(lr)
    }
}

impl From<Tensor<1>> for LearningRate {
    fn from(lr: Tensor<1>) -> Self {
        Self::Device(lr)
    }
}

impl From<Tensor<0>> for LearningRate {
    fn from(lr: Tensor<0>) -> Self {
        Self::Device(lr.reshape([1]))
    }
}

impl LearningRate {
    /// `tensor * self`: scale an update by the learning rate.
    pub fn apply<const D: usize>(&self, tensor: Tensor<D>) -> Tensor<D> {
        match self {
            Self::Host(lr) => tensor.mul_scalar(*lr),
            Self::Device(lr) => {
                let lr = Self::broadcast(lr, &tensor);
                tensor.mul(lr)
            }
        }
    }

    /// `tensor / self`.
    pub fn divide<const D: usize>(&self, tensor: Tensor<D>) -> Tensor<D> {
        match self {
            Self::Host(lr) => tensor.div_scalar(*lr),
            Self::Device(lr) => {
                let lr = Self::broadcast(lr, &tensor);
                tensor.div(lr)
            }
        }
    }

    /// The host value, if the learning rate lives on the host.
    pub fn host(&self) -> Option<HostLr> {
        match self {
            Self::Host(lr) => Some(*lr),
            Self::Device(_) => None,
        }
    }

    /// `self * value`.
    pub fn mul_scalar(&self, value: f64) -> Self {
        match self {
            Self::Host(lr) => Self::Host(lr * value),
            Self::Device(lr) => Self::Device(lr.clone().mul_scalar(value)),
        }
    }

    /// `self / value`.
    pub fn div_scalar(&self, value: f64) -> Self {
        match self {
            Self::Host(lr) => Self::Host(lr / value),
            Self::Device(lr) => Self::Device(lr.clone().div_scalar(value)),
        }
    }

    /// `self + value`.
    pub fn add_scalar(&self, value: f64) -> Self {
        match self {
            Self::Host(lr) => Self::Host(lr + value),
            Self::Device(lr) => Self::Device(lr.clone().add_scalar(value)),
        }
    }

    /// `value - self`.
    pub fn rsub_scalar(&self, value: f64) -> Self {
        match self {
            Self::Host(lr) => Self::Host(value - lr),
            Self::Device(lr) => Self::Device(lr.clone().neg().add_scalar(value)),
        }
    }

    /// `min(self, value)`.
    pub fn min_scalar(&self, value: f64) -> Self {
        match self {
            Self::Host(lr) => Self::Host(lr.min(value)),
            Self::Device(lr) => Self::Device(lr.clone().clamp_max(value)),
        }
    }

    /// Move a device learning rate next to the parameters it updates, leaving autodiff: the
    /// optimizer step runs on the inner tensors.
    pub fn to_device(self, device: &Device) -> Self {
        match self {
            Self::Host(lr) => Self::Host(lr),
            Self::Device(lr) => {
                let lr = lr.inner();
                if &lr.device() == device {
                    Self::Device(lr)
                } else {
                    Self::Device(lr.to_device(device))
                }
            }
        }
    }

    fn broadcast<const D: usize>(lr: &Tensor<1>, tensor: &Tensor<D>) -> Tensor<D> {
        lr.clone()
            .cast(FloatDType::from(tensor.dtype()))
            .reshape([1; D])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optim::decay::WeightDecayConfig;
    use crate::optim::momentum::MomentumConfig;
    use crate::{
        AdaGradConfig, AdafactorConfig, AdamConfig, AdamWConfig, AdanConfig, GradientsParams,
        LambConfig, LionConfig, ModuleOptimizer, MuonConfig, Optimizer, RmsPropConfig, SgdConfig,
    };
    use burn_core::tensor::{Distribution, Tolerance};
    use burn_nn::{Linear, LinearConfig};

    const LRS: [f64; 3] = [1e-2, 5e-3, 2e-2];

    /// Steps `optim` with a host and a device learning rate, changing it every step, and checks
    /// both follow the same trajectory.
    fn assert_device_lr_matches_host<O: Optimizer>(optim: O) {
        let device = Device::default();
        let mut host = Tensor::<2>::random([6, 4], Distribution::Default, &device);
        let mut on_device = host.clone();
        let (mut host_state, mut device_state) = (None, None);

        for (i, lr) in LRS.into_iter().enumerate() {
            let grad = Tensor::<2>::random([6, 4], Distribution::Default, &device)
                .add_scalar(i as f64 * 0.1);
            let device_lr = LearningRate::from(Tensor::<1>::from_floats([lr], &device));

            (host, host_state) = optim.step(lr.into(), host, grad.clone(), host_state);
            (on_device, device_state) = optim.step(device_lr, on_device, grad, device_state);

            on_device
                .to_data()
                .assert_approx_eq::<f32>(&host.to_data(), Tolerance::absolute(1e-6));
        }
    }

    #[test]
    fn device_lr_matches_host_sgd() {
        assert_device_lr_matches_host(
            SgdConfig::new()
                .with_weight_decay(Some(WeightDecayConfig::new(0.05)))
                .with_momentum(Some(MomentumConfig::new()))
                .build(),
        );
    }

    #[test]
    fn device_lr_matches_host_adam() {
        assert_device_lr_matches_host(AdamConfig::new().build());
    }

    #[test]
    fn device_lr_matches_host_adamw() {
        assert_device_lr_matches_host(AdamWConfig::new().with_weight_decay(0.1).build());
        assert_device_lr_matches_host(
            AdamWConfig::new()
                .with_weight_decay(0.1)
                .with_cautious_weight_decay(true)
                .build(),
        );
    }

    #[test]
    fn device_lr_matches_host_adagrad() {
        assert_device_lr_matches_host(AdaGradConfig::new().with_lr_decay(0.1).build());
    }

    #[test]
    fn device_lr_matches_host_adan() {
        assert_device_lr_matches_host(AdanConfig::new().with_weight_decay(0.1).build());
        assert_device_lr_matches_host(
            AdanConfig::new()
                .with_weight_decay(0.1)
                .with_no_prox(true)
                .build(),
        );
    }

    #[test]
    fn device_lr_matches_host_rmsprop() {
        assert_device_lr_matches_host(RmsPropConfig::new().with_centered(true).build());
    }

    #[test]
    fn device_lr_matches_host_muon() {
        assert_device_lr_matches_host(
            MuonConfig::new()
                .with_weight_decay(Some(WeightDecayConfig::new(0.1)))
                .build(),
        );
    }

    #[test]
    fn device_lr_matches_host_adafactor() {
        assert_device_lr_matches_host(AdafactorConfig::new().with_weight_decay(0.1).build());
        assert_device_lr_matches_host(
            AdafactorConfig::new()
                .with_relative_step(false)
                .with_scale_parameter(false)
                .build(),
        );
    }

    #[test]
    fn device_lr_matches_host_lamb() {
        assert_device_lr_matches_host(LambConfig::new().with_weight_decay(0.1).build());
    }

    #[test]
    fn device_lr_matches_host_lion() {
        assert_device_lr_matches_host(LionConfig::new().with_weight_decay(0.1).build());
    }

    #[test]
    fn module_optimizer_accepts_device_lr() {
        let device = Device::default().autodiff();
        let model = LinearConfig::new(4, 3).init(&device);
        let input = Tensor::<2>::random([5, 4], Distribution::Default, &device);
        let grads = |model: &Linear| {
            GradientsParams::from_grads(model.forward(input.clone()).sum().backward(), model)
        };
        let mut host_optim: ModuleOptimizer = AdamWConfig::new().init();
        let mut device_optim: ModuleOptimizer = AdamWConfig::new().init();
        let (mut host_model, mut device_model) = (model.clone(), model);

        for (i, lr) in LRS.into_iter().enumerate() {
            // The learning rate is an input like any other, autodiff device included, as a `[1]`
            // or a scalar tensor.
            let device_lr = Tensor::<1>::from_floats([lr], &device);
            let device_lr = match i % 2 {
                0 => LearningRate::from(device_lr),
                _ => LearningRate::from(device_lr.reshape::<0, _>([0usize; 0])),
            };
            host_model = host_optim.step(lr, host_model.clone(), grads(&host_model));
            device_model = device_optim.step(device_lr, device_model.clone(), grads(&device_model));
        }

        device_model.weight.val().to_data().assert_approx_eq::<f32>(
            &host_model.weight.val().to_data(),
            Tolerance::absolute(1e-6),
        );
    }
}
