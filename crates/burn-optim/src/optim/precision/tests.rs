use super::*;
use crate::{
    AdafactorConfig, AdamWConfig, ComputePrecision, GradientsParams, ModuleOptimizer, SgdConfig,
    StateStorage, grad_clipping::GradientClipping, momentum::MomentumConfig,
};
use burn::module::{Module, Param, ParamGroup};
use burn::tensor::DType;
use burn_core as burn;

fn tensor(value: f32, dtype: DType) -> Tensor<1> {
    Tensor::from_data([value], (&Device::default(), dtype))
}

fn value(tensor: &Tensor<1>) -> f32 {
    tensor.clone().into_scalar::<f32>()
}

fn close(actual: f32, expected: f32) {
    assert!((actual - expected).abs() < 2e-7, "{actual} != {expected}");
}

#[test]
fn master_weights_accumulate_updates_for_stateless_sgd() {
    let optimizer = SgdConfig::new()
        .build()
        .with_precision(OptimizerPrecisionConfig::new());
    let mut model = tensor(1.0, DType::BF16);
    let mut state = None;
    for _ in 0..64 {
        (model, state) = optimizer.step(1.0 / 8192.0, model, tensor(1.0, DType::BF16), state);
    }
    assert_eq!(model.dtype(), DType::BF16);
    assert_eq!(value(&model), 0.9921875);
    let master = state.unwrap().master.unwrap();
    assert_eq!(master.dtype(), DType::F32);
    assert_eq!(value(&master), 0.9921875);
}

#[test]
fn fresh_moments_are_used_before_storage_rounding() {
    let optimizer = AdamWConfig::new()
        .with_beta_1(0.5)
        .with_beta_2(0.5)
        .with_epsilon(0.0)
        .with_weight_decay(0.125)
        .build()
        .with_precision(
            OptimizerPrecisionConfig::new()
                .with_state_storage(StateStorage::Float(FloatDType::BF16)),
        );
    let (model, state) = optimizer.step(
        0.1,
        tensor(1.0, DType::BF16),
        tensor(0.30078125, DType::BF16),
        None,
    );
    let state = state.unwrap();
    // First bias-corrected moment quotient is one. Encoding the fresh second moment first
    // would perturb the quotient. Decay is applied to the authoritative parameter.
    close(value(state.master.as_ref().unwrap()), 0.8875);
    let moments = &state.algorithm.as_ref().unwrap().momentum;
    assert_eq!(moments.moment_1.dtype(), DType::BF16);
    assert_eq!(moments.moment_2.dtype(), DType::BF16);
    assert_eq!(f64::from(value(&moments.moment_2)), 0.045166015625);
    let (_, state) = optimizer.step(0.1, model, tensor(0.0, DType::BF16), Some(state));
    let state = state.unwrap();
    // Scalar reference uses the encoded moments from step one.
    let m_hat = (0.150390625_f64 * 0.5) / 0.75;
    let v_hat = (0.045166015625_f64 * 0.5) / 0.75;
    let expected = 0.8875 * (1.0 - 0.1 * 0.125) - 0.1 * m_hat / v_hat.sqrt();
    close(value(state.master.as_ref().unwrap()), expected as f32);
    assert_eq!(state.algorithm.unwrap().momentum.time, 2);
}

#[test]
fn optional_momentum_uses_the_same_storage_policy() {
    let optimizer = SgdConfig::new()
        .with_momentum(Some(
            MomentumConfig::new().with_momentum(0.5).with_dampening(0.0),
        ))
        .build()
        .with_precision(
            OptimizerPrecisionConfig::new()
                .with_state_storage(StateStorage::Float(FloatDType::BF16)),
        );
    let (model, state) =
        optimizer.step(0.1, tensor(1.0, DType::BF16), tensor(0.3, DType::F32), None);
    let (_, state) = optimizer.step(0.1, model, tensor(0.0, DType::F32), state);
    // The first velocity rounds to 0.30078125 in BF16, then decays by 0.5.
    close(
        value(state.unwrap().master.as_ref().unwrap()),
        (1.0_f64 - 0.03 - 0.0150390625) as f32,
    );
}

#[test]
fn amsgrad_maximum_retains_computation_precision() {
    let optimizer = AdamWConfig::new()
        .with_beta_1(0.5)
        .with_beta_2(0.5)
        .with_epsilon(0.0)
        .with_weight_decay(0.0)
        .with_amsgrad(true)
        .build()
        .with_precision(
            OptimizerPrecisionConfig::new()
                .with_state_storage(StateStorage::Float(FloatDType::BF16)),
        );
    let (model, state) = optimizer.step(
        0.1,
        tensor(1.0, DType::BF16),
        tensor(0.30078125, DType::BF16),
        None,
    );
    let (_, state) = optimizer.step(0.1, model, tensor(0.0, DType::BF16), state);
    let moments = state.unwrap().algorithm.unwrap().momentum;
    let maximum = moments.max_moment_2.unwrap();
    assert_eq!(maximum.dtype(), DType::F32);
    assert_eq!(f64::from(value(&maximum)), 0.04523468017578125);
    assert!(value(&maximum) > value(&moments.moment_2));
}

#[test]
fn factored_state_preserves_its_own_shapes() {
    let device = Device::default();
    let optimizer = AdafactorConfig::new()
        .with_relative_step(false)
        .with_scale_parameter(false)
        .build()
        .with_precision(
            OptimizerPrecisionConfig::new()
                .with_state_storage(StateStorage::Float(FloatDType::BF16)),
        );
    let parameter = Tensor::<2>::ones([2, 2], (&device, DType::BF16));
    let grad = Tensor::from_data([[0.125, 0.25], [0.5, 1.0]], (&device, DType::F32));
    let (_, state) = optimizer.step(0.01, parameter, grad, None);
    let state = state.unwrap().algorithm.unwrap();
    assert_eq!(state.second_moment.dtype(), DType::BF16);
    assert_eq!(state.second_moment.dims(), [2, 1]);
    let column = state.column_second_moment.unwrap();
    assert_eq!(column.dtype(), DType::BF16);
    assert_eq!(column.dims(), [1, 2]);
}

#[test]
fn default_promotion_preserves_f64_without_a_master_copy() {
    let device = Device::default();
    if !device.supports_dtype(FloatDType::F64) {
        return;
    }
    let optimizer = SgdConfig::new()
        .build()
        .with_precision(OptimizerPrecisionConfig::new());
    let parameter = Tensor::<1>::from_data([1.0_f64], (&device, DType::F64));
    let grad = Tensor::<1>::from_data([1e-12_f64], (&device, DType::F64));
    let (model, state) = optimizer.step(1.0, parameter, grad, None);
    assert_eq!(model.dtype(), DType::F64);
    assert_eq!(model.into_scalar::<f64>(), 1.0 - 1e-12);
    assert!(state.unwrap().master.is_none());
}

#[test]
fn direct_updates_do_not_retain_master_weights() {
    let optimizer = SgdConfig::new().build().with_precision(
        OptimizerPrecisionConfig::new().with_parameter_update(ParameterUpdate::Direct),
    );
    let (model, state) = optimizer.step(
        1.0 / 8192.0,
        tensor(1.0, DType::BF16),
        tensor(1.0, DType::F32),
        None,
    );
    assert_eq!(value(&model), 1.0);
    assert!(state.unwrap().master.is_none());
}

#[test]
#[should_panic(expected = "Optimizer precision differs from its saved state")]
fn changing_policy_requires_resetting_state() {
    let optimizer = SgdConfig::new()
        .build()
        .with_precision(OptimizerPrecisionConfig::new());
    let (model, state) =
        optimizer.step(0.1, tensor(1.0, DType::BF16), tensor(1.0, DType::F32), None);
    let changed = SgdConfig::new()
        .build()
        .with_precision(OptimizerPrecisionConfig::new().with_compute(ComputePrecision::Parameter));
    let _ = changed.step(0.1, model, tensor(1.0, DType::F32), state);
}

#[derive(Module, Debug)]
struct Pair {
    accumulated: Param<Tensor<1>>,
    direct: Param<Tensor<1>>,
}

fn pair() -> Pair {
    let device = Device::default().autodiff();
    Pair {
        accumulated: Param::from_tensor(
            Tensor::from_data([1.0], (&device, DType::BF16)).require_grad(),
        ),
        direct: Param::from_tensor(Tensor::from_data([1.0], (&device, DType::BF16)).require_grad()),
    }
}

#[test]
fn parameter_groups_and_reset_include_master_weights() {
    let mut model = pair();
    let group = ParamGroup::ids_from_module(model.direct.clone());
    let mut optimizer = ModuleOptimizer::from(
        SgdConfig::new()
            .build()
            .with_precision(OptimizerPrecisionConfig::new()),
    )
    .with_group(group, SgdConfig::new().build(), None);
    for _ in 0..32 {
        let mut grads = GradientsParams::new();
        grads.register(model.accumulated.id, tensor(1.0, DType::BF16));
        grads.register(model.direct.id, tensor(1.0, DType::BF16));
        model = optimizer.step(1.0 / 8192.0, model, grads);
    }
    assert_eq!(value(&model.accumulated.val()), 0.99609375);
    assert_eq!(value(&model.direct.val()), 1.0);
    assert!(model.accumulated.val().is_require_grad());

    // Replacing model data while keeping its identity requires discarding old masters.
    let id = model.accumulated.id;
    model.accumulated = Param::initialized(
        id,
        Tensor::from_data([2.0], (&Device::default().autodiff(), DType::BF16)).require_grad(),
    );
    optimizer.reset_state();
    assert!(optimizer.to_record().is_empty());
    let mut grads = GradientsParams::new();
    grads.register(id, tensor(0.0, DType::BF16));
    model = optimizer.step(0.1, model, grads);
    assert_eq!(value(&model.accumulated.val()), 2.0);
}

#[test]
fn gradient_conversion_precedes_module_clipping() {
    let model = pair();
    let mut optimizer = ModuleOptimizer::from(
        SgdConfig::new()
            .build()
            .with_precision(OptimizerPrecisionConfig::new()),
    )
    .with_grad_clipping(GradientClipping::Value(1e-8));
    let mut grads = GradientsParams::new();
    // F16 cannot represent the clipping threshold. Clipping before promotion would erase it.
    grads.register(model.accumulated.id, tensor(1.0, DType::F16));
    let model = optimizer.step(1e7, model, grads);
    assert_eq!(value(&model.accumulated.val()), 0.8984375);
}

#[test]
fn checkpoint_continuation_preserves_moments_and_master() {
    let mut model = pair();
    let make_optimizer = || {
        ModuleOptimizer::from(
            AdamWConfig::new()
                .with_beta_1(0.5)
                .with_beta_2(0.5)
                .with_weight_decay(0.0)
                .build()
                .with_precision(
                    OptimizerPrecisionConfig::new()
                        .with_state_storage(StateStorage::Float(FloatDType::BF16)),
                ),
        )
    };
    let mut original = make_optimizer();
    let grads = |model: &Pair, g| {
        let mut grads = GradientsParams::new();
        grads.register(model.accumulated.id, tensor(g, DType::F32));
        grads
    };
    for _ in 0..3 {
        let gradients = grads(&model, 1e-4);
        model = original.step(1e-4, model, gradients);
    }
    let mut restored = make_optimizer()
        .from_bytes(original.clone().into_bytes().unwrap())
        .unwrap();
    let expected = original.step(0.01, model.clone(), grads(&model, -2e-4));
    let actual = restored.step(0.01, model.clone(), grads(&model, -2e-4));
    expected
        .accumulated
        .to_data()
        .assert_eq(&actual.accumulated.to_data(), true);
    let expected = original.to_record();
    for tensor in restored.to_record().tensors {
        let other = expected
            .tensors
            .iter()
            .find(|other| other.name == tensor.name)
            .unwrap();
        tensor.data.assert_eq(&other.data, true);
    }
}
