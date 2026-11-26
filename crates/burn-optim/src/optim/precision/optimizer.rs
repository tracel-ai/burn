use super::{ComputePrecision, state::StoredState};
use crate::{LearningRate, Optimizer, OptimizerStateStorage};
use burn_core::tensor::{Device, FloatDType, Tensor};

#[derive(Clone)]
pub(super) struct PrecisionOptimizer<O, S> {
    optimizer: O,
    compute: FloatDType,
    storage: S,
}

impl<O: Optimizer, S> PrecisionOptimizer<O, S> {
    pub(super) fn new(optimizer: O, compute: FloatDType, storage: S) -> Self {
        Self {
            optimizer: optimizer.without_default_precision(),
            compute,
            storage,
        }
    }
}

impl<O: Optimizer, S: OptimizerStateStorage<O>> Optimizer for PrecisionOptimizer<O, S> {
    type State<const D: usize> = StoredState<S::Stored<D>>;

    fn prepare_gradient<const D: usize>(&self, grad: Tensor<D>) -> Tensor<D> {
        self.optimizer.prepare_gradient(grad).cast(self.compute)
    }

    fn step<const D: usize>(
        &self,
        lr: LearningRate,
        model: Tensor<D>,
        grad: Tensor<D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<D>, Option<Self::State<D>>) {
        let statistics = state.map(|state| self.storage.restore::<D>(state.0, self.compute));
        let (updated, statistics) =
            ComputePrecision::Fixed(self.compute).apply(model, grad, |model, grad| {
                self.optimizer.step(lr, model, grad, statistics)
            });
        (
            updated,
            statistics.map(|s| StoredState(self.storage.store::<D>(s))),
        )
    }

    fn to_device<const D: usize>(state: Self::State<D>, device: &Device) -> Self::State<D> {
        StoredState(S::to_device::<D>(state.0, device))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AdamConfig, AdamWConfig, GradientsParams, LionConfig, ModuleOptimizer, SgdConfig,
        grad_clipping::GradientClipping,
    };
    use crate::{MappableOptimizer, OptimizerPrecision, OptimizerStateMapper};
    use burn_core::module::{Param, ParamGroup};
    use burn_core::tensor::{DType, FloatDType};

    const DIRECT: OptimizerPrecision = OptimizerPrecision::new(FloatDType::F32);

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
        let sgd = SgdConfig::new().build();
        let mut optim = ModuleOptimizer::from(DIRECT.wrap(sgd.clone())).with_group(
            ParamGroup::from_path("1"),
            DIRECT.wrap_with_master_weights(sgd),
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
        let policy = DIRECT.with_state_storage(FloatDType::BF16);
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
        let policy = OptimizerPrecision::new(FloatDType::F16).with_state_storage(FloatDType::F32);
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

        let mut optimizer = ModuleOptimizer::from(DIRECT.wrap_with_master_weights(config.build()))
            .load_record(ordinary.to_record());
        let param = update(&mut optimizer, param, 0.0625, -1.0 / 2048.0);
        assert_eq!(param.val().into_scalar::<f32>(), 0.9604492);
    }
    #[derive(Clone)]
    struct MomentStorageMapper;
    impl OptimizerStateMapper for MomentStorageMapper {
        fn map<const D: usize>(&self, tensor: Tensor<D>, path: &crate::StatePath<'_>) -> Tensor<D> {
            if path.matches(&["momentum", "moment_1"]) {
                tensor.cast(FloatDType::BF16)
            } else {
                tensor
            }
        }
    }

    #[test]
    fn component_storage_can_keep_second_moments_in_compute_precision() {
        let mut optimizer = ModuleOptimizer::from(
            DIRECT
                .with_state_storage(MomentStorageMapper)
                .wrap(AdamConfig::new().build()),
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
        let mut optimizer =
            ModuleOptimizer::from(DIRECT.wrap_with_master_weights(SgdConfig::new().build()))
                .with_grad_clipping(GradientClipping::Value(1e-8));
        let param = update(&mut optimizer, parameter(1.0, FloatDType::F16), 1e7, 1.0);
        assert_eq!(param.val().into_scalar::<f32>(), 0.89990234);
    }

    #[test]
    fn bare_checkpoint_can_resume_in_device_default_precision() {
        let config = AdamWConfig::new()
            .with_beta_1(0.5)
            .with_beta_2(0.5)
            .with_epsilon(0.5)
            .with_weight_decay(0.0);
        let mut original = config.init();
        let param = update(&mut original, parameter(1.0, FloatDType::F16), 0.0, 1.0)
            .map(|t| t.cast(FloatDType::F32));
        let mut restored = config.init().load_record(original.to_record());
        let record = restored.to_record();
        assert_eq!(record.len(), 2);
        for leaf in [".momentum.moment_1", ".momentum.moment_2"] {
            assert!(
                record
                    .tensors
                    .iter()
                    .any(|t| t.name.ends_with(leaf) && t.data.dtype() == DType::F32)
            );
        }

        // After gradients [1, 0], both bias-corrected moments are 1/3.
        // A fresh optimizer with a zero gradient would leave the parameter at 1.
        let param = update(&mut restored, param, 1.0, 0.0);
        let moment = 1.0_f32 / 3.0;
        let expected = 1.0 - moment / (moment.sqrt() + 0.5);
        assert!((param.val().into_scalar::<f32>() - expected).abs() < 1e-6);
    }

    #[test]
    fn reset_discards_master_after_model_replacement() {
        let mut optimizer =
            ModuleOptimizer::from(DIRECT.wrap_with_master_weights(SgdConfig::new().build()));
        let param = update(&mut optimizer, parameter(1.0, FloatDType::F16), 0.01, 1.0);
        let param = param.map(|t| t.ones_like().mul_scalar(2.0));
        optimizer.reset_state();
        let param = update(&mut optimizer, param, 0.01, 0.0);
        assert_eq!(param.val().into_scalar::<f32>(), 2.0);
    }
    #[derive(Clone)]
    struct StatelessOptimizer;
    impl Optimizer for StatelessOptimizer {
        fn prepare_gradient<const D: usize>(&self, grad: Tensor<D>) -> Tensor<D> {
            grad.mul_scalar(2.)
        }
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
    impl MappableOptimizer for StatelessOptimizer {
        fn map_state<const D: usize, M: OptimizerStateMapper>(_: (), _: &M) {}
    }

    #[test]
    fn master_only_checkpoint_accumulates_unrounded_updates() {
        let mut optimizer =
            ModuleOptimizer::from(DIRECT.wrap_with_master_weights(StatelessOptimizer));
        let param = update(
            &mut optimizer,
            parameter(1.0, FloatDType::F16),
            1.0 / 16384.0,
            1.0,
        );
        let mut restored =
            ModuleOptimizer::from(DIRECT.wrap_with_master_weights(StatelessOptimizer))
                .from_bytes(optimizer.into_bytes().unwrap())
                .unwrap();
        let param = update(&mut restored, param, 1.0 / 16384.0, 1.0);
        let param = update(&mut restored, param, 1.0 / 16384.0, 1.0);
        assert_eq!(param.val().into_scalar::<f32>(), 1.0 - 1.0 / 2048.0);
    }
}
