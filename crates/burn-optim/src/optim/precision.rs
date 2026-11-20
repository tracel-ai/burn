use burn_core::tensor::{Device, FloatDType, Tensor};
use serde::{Deserialize, Serialize};

use crate::{LearningRate, Optimizer, RecordState, StateSink, StateSource};

/// Precision for a per-parameter update on unquantized floating-point parameters and gradients.
/// The update device must support the selected computation and storage dtypes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptimizerPrecision {
    /// Dtype of the parameter, gradient and decoded statistics supplied to the equation.
    pub compute: FloatDType,
    /// Keep the unrounded update for the next step when model and compute dtypes differ.
    pub master_weights: bool,
    /// Representation of statistics between updates. Scalar counters keep their original types.
    pub state: OptimizerStateStorage,
}

/// How the optimizer retains its tensor statistics between calls to `step`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizerStateStorage {
    /// Round each updated statistic to the given floating-point dtype on its device.
    Float(FloatDType),
}

impl OptimizerStateStorage {
    fn encode<const D: usize>(self, tensor: Tensor<D>) -> Tensor<D> {
        match self {
            Self::Float(dtype) => tensor.cast(dtype),
        }
    }
}

impl OptimizerPrecision {
    /// Apply this policy to a bare optimizer for use with a module or parameter group.
    /// Module gradient clipping runs before the update, in the incoming gradient's dtype
    /// (with F32 norm accumulation for half-precision gradients).
    pub fn wrap<O: PrecisionConfigurable>(self, optimizer: O) -> impl Optimizer {
        PrecisionOptimizer {
            optimizer,
            precision: self,
        }
    }
}

/// An optimizer whose floating-point statistics can be transformed on device.
/// Its state record must reserve the top-level name `__precision` for the wrapper.
/// Derive [`crate::OptimizerRecordState`] for records with tensor leaves, including nested
/// states, or use [`StateSource::tensor_options`] when restoring those leaves by hand.
pub trait PrecisionConfigurable: Optimizer {
    /// Apply `map` to every statistics tensor, preserving its shape and scalar bookkeeping.
    fn map_state<const D: usize>(
        state: Self::State<D>,
        map: impl Fn(Tensor<D>) -> Tensor<D>,
    ) -> Self::State<D>;

    /// Apply the equation in the supplied tensors' dtype. Override when the ordinary
    /// step selects a different dtype internally.
    fn step_in_dtype<const D: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<D>,
        grad: Tensor<D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<D>, Option<Self::State<D>>) {
        self.step(lr, tensor, grad, state)
    }
}

#[derive(Clone)]
struct PrecisionOptimizer<O> {
    optimizer: O,
    precision: OptimizerPrecision,
}

const MASTER_KEY: &str = "__precision.master";
const STATELESS_KEY: &str = "__precision.stateless";

#[derive(Clone)]
struct PrecisionState<S, const D: usize> {
    pub statistics: Option<S>,
    pub master: Option<Tensor<D>>,
}

impl<S: RecordState, const D: usize> RecordState for PrecisionState<S, D> {
    fn state_flatten(&self, prefix: &str, out: &mut StateSink) {
        match &self.statistics {
            Some(state) => state.state_flatten(prefix, out),
            None => out.push_scalar(prefix, STATELESS_KEY, true.into()),
        }
        if let Some(master) = &self.master {
            out.push_tensor(prefix, MASTER_KEY, master.to_data());
        }
    }

    fn state_unflatten(prefix: &str, src: &mut StateSource, device: &Device) -> Option<Self> {
        src.with_recorded_dtypes(|src| Self::restore(prefix, src, device))
    }
}

impl<S: RecordState, const D: usize> PrecisionState<S, D> {
    fn restore(prefix: &str, src: &mut StateSource, device: &Device) -> Option<Self> {
        let master = src.take_tensor(prefix, MASTER_KEY).map(|data| {
            let dtype = data.dtype();
            Tensor::from_data(data, (device, dtype))
        });
        if src.take_scalar(prefix, STATELESS_KEY).is_some() {
            return master.map(|master| Self {
                master: Some(master),
                statistics: None,
            });
        }
        Some(Self {
            statistics: Some(S::state_unflatten(prefix, src, device)?),
            master,
        })
    }
}

impl<O: PrecisionConfigurable> Optimizer for PrecisionOptimizer<O> {
    type State<const D: usize> = PrecisionState<O::State<D>, D>;

    fn step<const D: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<D>,
        grad: Tensor<D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<D>, Option<Self::State<D>>) {
        let model_dtype = tensor.dtype();
        let policy = self.precision;
        let keep_master = policy.master_weights && model_dtype != policy.compute.into();
        let PrecisionState { master, statistics } = state.unwrap_or(PrecisionState {
            master: None,
            statistics: None,
        });
        let parameter = master
            .filter(|_| policy.master_weights)
            .unwrap_or(tensor)
            .cast(policy.compute);
        let statistics = statistics.map(|state| O::map_state(state, |t| t.cast(policy.compute)));
        let (updated, statistics) =
            self.optimizer
                .step_in_dtype(lr, parameter, grad.cast(policy.compute), statistics);
        let statistics = statistics.map(|state| O::map_state(state, |t| policy.state.encode(t)));
        let master = keep_master.then(|| updated.clone());
        let state = (statistics.is_some() || master.is_some())
            .then_some(PrecisionState { statistics, master });
        (updated.cast(model_dtype), state)
    }

    fn to_device<const D: usize>(state: Self::State<D>, device: &Device) -> Self::State<D> {
        PrecisionState {
            statistics: state.statistics.map(|s| O::to_device::<D>(s, device)),
            master: state.master.map(|m| m.to_device(device)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AdamConfig, AdamWConfig, GradientsParams, LionConfig, ModuleOptimizer, SgdConfig,
        grad_clipping::GradientClipping,
    };
    use burn_core::module::{Param, ParamGroup};
    use burn_core::tensor::DType;

    const DIRECT: OptimizerPrecision = OptimizerPrecision {
        compute: FloatDType::F32,
        master_weights: false,
        state: OptimizerStateStorage::Float(FloatDType::F32),
    };

    fn parameter(value: f32, dtype: FloatDType) -> Param<Tensor<1>> {
        let device = Device::default().autodiff();
        Param::from_tensor(Tensor::from_floats([value], &device).cast(dtype))
    }

    fn gradients(param: &Param<Tensor<1>>, value: f32) -> GradientsParams {
        let mut grads = GradientsParams::new();
        let grad = Tensor::<1>::from_floats([value], &param.val().device().inner())
            .cast(param.val().dtype());
        grads.register(param.id, grad);
        grads
    }

    fn update(
        optim: &mut ModuleOptimizer,
        param: Param<Tensor<1>>,
        lr: f64,
        grad: f32,
    ) -> Param<Tensor<1>> {
        let grads = gradients(&param, grad);
        optim.step(lr, param, grads)
    }

    // The first squared gradient fits in F16, but its second-moment contribution
    // rounds to zero. Reverse and double the gradient to exercise the saved moments.
    fn check_small_gradients(mut optimizer: ModuleOptimizer, expected: [f32; 2]) {
        let mut param = parameter(1.0, FloatDType::F16);
        for (grad, expected) in [1.0 / 4096.0, -1.0 / 2048.0].into_iter().zip(expected) {
            param = update(&mut optimizer, param, 0.0625, grad);
            assert_eq!(param.val().dtype(), DType::F16);
            assert_eq!(param.val().into_scalar::<f32>(), expected);
        }
    }

    #[test]
    fn adam_small_f16_gradients_follow_bias_corrected_updates() {
        let adam = AdamConfig::new().with_epsilon(1e-8).build();
        check_small_gradients(DIRECT.wrap(adam).into(), [0.9375, 0.9604492]);
    }

    #[test]
    fn adamw_small_f16_gradients_follow_decoupled_decay_updates() {
        let adamw = AdamWConfig::new()
            .with_epsilon(1e-8)
            .with_weight_decay(0.25)
            .build();
        check_small_gradients(DIRECT.wrap(adamw).into(), [0.921875, 0.9301758]);
    }

    #[test]
    fn only_master_group_accumulates_sub_rounding_steps() {
        let master = OptimizerPrecision {
            master_weights: true,
            ..DIRECT
        };
        let sgd = SgdConfig::new().build();
        let mut optim = ModuleOptimizer::from(DIRECT.wrap(sgd.clone())).with_group(
            ParamGroup::from_path("1"),
            master.wrap(sgd),
            Some(GradientClipping::Value(0.5)),
        );
        let mut params: [_; 2] = core::array::from_fn(|_| parameter(1.0, FloatDType::F16));
        for _ in 0..4 {
            let mut grads = gradients(&params[0], 0.5);
            grads.register(
                params[1].id,
                Tensor::<1>::from_floats([2.0], &Device::default()).cast(FloatDType::F16),
            );
            params = optim.step(1.0 / 4096.0, params, grads);
        }
        assert_eq!(params[0].val().into_scalar::<f32>(), 1.0);
        assert_eq!(params[1].val().into_scalar::<f32>(), 1.0 - 1.0 / 2048.0);
        let record = optim.to_record();
        assert_eq!(record.len(), 1);
        assert_eq!(record.tensors[0].param_id, Some(params[1].id.val()));
        assert_eq!(record.tensors[0].data.dtype(), DType::F32);
    }

    #[test]
    fn next_update_uses_rounded_statistics_with_f32_parameters() {
        let lion = LionConfig::new().with_beta_1(0.5).with_beta_2(0.5).build();
        let policy = OptimizerPrecision {
            state: OptimizerStateStorage::Float(FloatDType::BF16),
            ..DIRECT
        };
        let mut optim = ModuleOptimizer::from(policy.wrap(lion));
        let param = update(&mut optim, parameter(0.0, FloatDType::F32), 1.0, 1.003);
        let record = optim.to_record();
        assert_eq!(record.len(), 1);
        assert_eq!(record.tensors[0].data.dtype(), DType::BF16);
        // BF16 rounds the stored momentum from 0.5015 to 0.5. These give
        // opposite signs when interpolated with the next gradient.
        let param = update(&mut optim, param, 1.0, -0.50075);
        assert_eq!(param.val().into_scalar::<f32>(), 0.0);
    }

    #[test]
    fn explicit_compute_overrides_adafactor_half_promotion() {
        let policy = OptimizerPrecision {
            compute: FloatDType::F16,
            ..DIRECT
        };
        let mut optimizer =
            ModuleOptimizer::from(policy.wrap(crate::AdafactorConfig::new().build()));
        let _ = update(
            &mut optimizer,
            parameter(1.0, FloatDType::F32),
            0.01,
            1.0 + 1.0 / 1024.0,
        );
        let record = optimizer.to_record();
        let moment = &record.tensors[0].data;
        // The square in F32 also includes 2^-20, which the F16 equation drops.
        assert_eq!(moment.as_slice::<f32>().unwrap(), &[1.0 + 1.0 / 512.0]);
    }

    #[test]
    fn ordinary_checkpoint_supplies_moments_for_first_master_update() {
        let config = AdamConfig::new().with_epsilon(1e-8);
        let mut ordinary = config.init();
        let param = update(
            &mut ordinary,
            parameter(1.0, FloatDType::F32),
            0.0625,
            1.0 / 4096.0,
        )
        .map(|tensor| tensor.cast(FloatDType::F16));
        let policy = OptimizerPrecision {
            master_weights: true,
            ..DIRECT
        };
        let mut optimizer =
            ModuleOptimizer::from(policy.wrap(config.build())).load_record(ordinary.to_record());
        let param = update(&mut optimizer, param, 0.0625, -1.0 / 2048.0);
        assert_eq!(param.val().into_scalar::<f32>(), 0.9604492);
    }
}
