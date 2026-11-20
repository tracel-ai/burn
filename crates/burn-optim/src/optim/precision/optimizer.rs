use super::{compute_mapper::ComputeOptimizerStateMapper, state::PrecisionState};
use crate::{
    ConfigurableOptimizer, LearningRate, Optimizer, OptimizerPrecision, OptimizerStateMapper,
};
use burn_core::tensor::{Device, Tensor};

#[derive(Clone)]
pub(super) struct PrecisionOptimizer<O, M> {
    optimizer: O,
    precision: OptimizerPrecision,
    storage: M,
}

impl<O, M> PrecisionOptimizer<O, M> {
    pub(super) fn new(optimizer: O, precision: OptimizerPrecision, storage: M) -> Self {
        Self {
            optimizer,
            precision,
            storage,
        }
    }
}

impl<O: ConfigurableOptimizer, M: OptimizerStateMapper + Clone + Send + Sync + 'static> Optimizer
    for PrecisionOptimizer<O, M>
{
    type State<const D: usize> = PrecisionState<O::State<D>, D>;

    fn prepare_gradient<const D: usize>(&self, grad: Tensor<D>) -> Tensor<D> {
        grad.cast(self.precision.compute)
    }

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
        let statistics = statistics.map(|state| {
            O::map_state(state, &mut ComputeOptimizerStateMapper::new(policy.compute))
        });
        let (updated, statistics) =
            self.optimizer
                .step_in_dtype(lr, parameter, grad.cast(policy.compute), statistics);
        let statistics = statistics.map(|state| O::map_state(state, &mut self.storage.clone()));
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
    use crate::OptimizerStateStorage;
    use crate::{
        AdamConfig, AdamWConfig, GradientsParams, LionConfig, ModuleOptimizer, SgdConfig,
        grad_clipping::GradientClipping,
    };
    use burn_core::module::{Param, ParamGroup};
    use burn_core::tensor::{DType, FloatDType};

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
    #[test]
    fn quantized_statistics_drive_the_next_update_after_checkpoint() {
        use burn_core::tensor::quantization::{QuantScheme, QuantStore, QuantValue};
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_store(QuantStore::Native);
        let policy = OptimizerPrecision {
            state: OptimizerStateStorage::Quantized(scheme),
            ..DIRECT
        };
        let create = || {
            ModuleOptimizer::from(
                policy.wrap(LionConfig::new().with_beta_1(0.5).with_beta_2(0.5).build()),
            )
        };
        let param = Param::from_tensor(Tensor::<1>::from_floats(
            [0.0, 0.0],
            &Device::default().autodiff(),
        ));
        let mut optimizer = create();
        let mut grads = GradientsParams::new();
        grads.register(
            param.id,
            Tensor::<1>::from_floats([1.0, 0.003], &Device::default()),
        );
        let param = optimizer.step(1.0, param, grads);
        assert!(matches!(
            optimizer.to_record().tensors[0].data.dtype(),
            DType::QFloat(_)
        ));
        let mut restored = create()
            .from_bytes(optimizer.into_bytes().unwrap())
            .unwrap();
        let mut grads = GradientsParams::new();
        grads.register(
            param.id,
            Tensor::<1>::from_floats([0.0, -0.001], &Device::default()),
        );
        let param = restored.step(1.0, param, grads);
        assert_eq!(param.val().try_into_vec_as::<f32>().unwrap(), [-2.0, 0.0]);
    }

    #[derive(Clone)]
    struct MomentStorageMapper;
    impl OptimizerStateMapper for MomentStorageMapper {
        fn map<const D: usize>(&mut self, tensor: Tensor<D>, name: &str) -> Tensor<D> {
            match name {
                "AdaptiveMomentumState.moment_1" => tensor.cast(FloatDType::BF16),
                _ => tensor,
            }
        }
    }

    #[test]
    fn component_storage_can_keep_second_moments_in_compute_precision() {
        let mut optimizer = ModuleOptimizer::from(
            DIRECT.wrap_with_storage(AdamConfig::new().build(), MomentStorageMapper),
        );
        let _ = update(
            &mut optimizer,
            parameter(1.0, FloatDType::F16),
            0.01,
            1.0 / 4096.0,
        );
        let record = optimizer.to_record();
        let first = record
            .tensors
            .iter()
            .find(|t| t.name.ends_with("moment_1"))
            .unwrap();
        let second = record
            .tensors
            .iter()
            .find(|t| t.name.ends_with("moment_2"))
            .unwrap();
        assert_eq!(first.data.dtype(), DType::BF16);
        assert_eq!(second.data.dtype(), DType::F32);
        assert!(second.data.as_slice::<f32>().unwrap()[0] > 0.0);
    }

    #[test]
    fn promotion_precedes_clipping() {
        let policy = OptimizerPrecision {
            master_weights: true,
            ..DIRECT
        };
        let mut optimizer = ModuleOptimizer::from(policy.wrap(SgdConfig::new().build()))
            .with_grad_clipping(GradientClipping::Value(1e-8));
        let param = update(&mut optimizer, parameter(1.0, FloatDType::F16), 1e7, 1.0);
        assert_eq!(param.val().into_scalar::<f32>(), 0.89990234);
    }

    #[test]
    fn bare_checkpoint_can_resume_in_device_default_precision() {
        let config = AdamWConfig::new().with_epsilon(0.001);
        let mut original = config.init();
        let param = update(&mut original, parameter(1.0, FloatDType::F16), 0.01, 1.0)
            .map(|t| t.cast(FloatDType::F32));
        let mut restored = config.init().load_record(original.to_record());
        assert!(
            restored
                .to_record()
                .tensors
                .iter()
                .all(|t| t.data.dtype() == DType::F32)
        );
        let before = param.val().into_scalar::<f32>();
        let param = update(&mut restored, param, 0.01, 1.0);
        assert!(param.val().into_scalar::<f32>() < before);
    }

    #[test]
    fn reset_discards_master_after_model_replacement() {
        let policy = OptimizerPrecision {
            master_weights: true,
            ..DIRECT
        };
        let mut optimizer = ModuleOptimizer::from(policy.wrap(SgdConfig::new().build()));
        let param = update(&mut optimizer, parameter(1.0, FloatDType::F16), 0.01, 1.0);
        let param = param.map(|t| t.ones_like().mul_scalar(2.0));
        optimizer.reset_state();
        let param = update(&mut optimizer, param, 0.01, 0.0);
        assert_eq!(param.val().into_scalar::<f32>(), 2.0);
    }
    #[derive(Clone)]
    struct StatelessOptimizer;
    impl Optimizer for StatelessOptimizer {
        type State<const D: usize> = ();
        fn step<const D: usize>(
            &self,
            lr: LearningRate,
            tensor: Tensor<D>,
            grad: Tensor<D>,
            _: Option<()>,
        ) -> (Tensor<D>, Option<()>) {
            (tensor - grad.mul_scalar(lr), None)
        }
        fn to_device<const D: usize>(_: (), _: &Device) {}
    }
    impl ConfigurableOptimizer for StatelessOptimizer {
        fn map_state<const D: usize, M: OptimizerStateMapper>(_: (), _: &mut M) {}
    }

    #[test]
    fn master_only_checkpoint_accumulates_unrounded_updates() {
        let policy = OptimizerPrecision {
            master_weights: true,
            ..DIRECT
        };
        let mut optimizer = ModuleOptimizer::from(policy.wrap(StatelessOptimizer));
        let param = update(
            &mut optimizer,
            parameter(1.0, FloatDType::F16),
            1.0 / 8192.0,
            1.0,
        );
        let mut restored = ModuleOptimizer::from(policy.wrap(StatelessOptimizer))
            .from_bytes(optimizer.into_bytes().unwrap())
            .unwrap();
        let param = update(&mut restored, param, 1.0 / 8192.0, 1.0);
        let param = update(&mut restored, param, 1.0 / 8192.0, 1.0);
        assert_eq!(param.val().into_scalar::<f32>(), 1.0 - 1.0 / 2048.0);
    }
}
