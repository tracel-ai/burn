use super::*;
use crate::GradientsParams;
use burn::{
    module::Param,
    tensor::{DType, FloatDType, TensorData, Tolerance},
};
use burn_nn::Linear;

#[test]
fn vector_matches_scalar_reference_for_two_steps() {
    let device = Device::default();
    let optimizer = AdafactorConfig::new()
        .with_epsilon_1(0.01)
        .with_clip_threshold(0.75)
        .with_relative_step(false)
        .with_scale_parameter(false)
        .build();
    let tensor = Tensor::<1>::from_floats([1.0, -2.0, 3.0], &device);

    // References evaluated with scalar f64 arithmetic, including update RMS clipping.
    let (tensor, state) = optimizer.step(
        0.2,
        tensor,
        Tensor::<1>::from_floats([0.2, -0.4, 0.7], &device),
        None,
    );
    tensor.to_data().assert_approx_eq::<f32>(
        &TensorData::from([0.859_130_0, -1.847_205_0, 2.844_085_5]),
        Tolerance::absolute(1e-6),
    );

    let (tensor, state) = optimizer.step(
        0.2,
        tensor,
        Tensor::<1>::from_floats([-0.1, 0.8, -0.5], &device),
        state,
    );
    tensor.to_data().assert_approx_eq::<f32>(
        &TensorData::from([0.951_171_5, -2.046_866_2, 2.982_519]),
        Tolerance::absolute(1e-6),
    );
    let state = state.unwrap();
    assert_eq!(state.time, 2);
    assert_eq!(state.second_moment.dims(), [3]);
    assert!(state.column_second_moment.is_none());
    state.second_moment.to_data().assert_approx_eq::<f32>(
        &TensorData::from([0.032_769_524, 0.445_687_6, 0.362_156_2]),
        Tolerance::absolute(1e-7),
    );
}

#[test]
fn matrix_matches_scalar_reference_for_two_steps() {
    let device = Device::default();
    let optimizer = AdafactorConfig::new()
        .with_epsilon_1(0.01)
        .with_clip_threshold(0.75)
        .with_relative_step(false)
        .with_scale_parameter(false)
        .build();
    let tensor = Tensor::<2>::from_floats([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]], &device);

    let (tensor, state) = optimizer.step(
        0.2,
        tensor,
        Tensor::<2>::from_floats([[0.2, -0.4, 0.6], [0.8, -1.0, 1.2]], &device),
        None,
    );
    tensor.to_data().assert_approx_eq::<f32>(
        &TensorData::from([
            [0.906_097_23, -1.855_350_6, 2.825_291_9],
            [-4.163_600_4, 5.157_508, -6.152_191],
        ]),
        Tolerance::absolute(1e-6),
    );

    let (tensor, state) = optimizer.step(
        0.2,
        tensor,
        Tensor::<2>::from_floats([[-0.5, 0.25, 1.0], [0.125, -0.75, 0.375]], &device),
        state,
    );
    tensor.to_data().assert_approx_eq::<f32>(
        &TensorData::from([
            [1.109_898, -1.929_492, 2.594_519_9],
            [-4.202_498, 5.327_315_3, -6.218_259],
        ]),
        Tolerance::absolute(1e-6),
    );
    let state = state.unwrap();
    assert_eq!(state.time, 2);
    assert_eq!(state.second_moment.dims(), [2, 1]);
    state.second_moment.to_data().assert_approx_eq::<f32>(
        &TensorData::from([[0.340_732_57], [0.584_606]]),
        Tolerance::absolute(1e-7),
    );
    let column = state.column_second_moment.unwrap();
    assert_eq!(column.dims(), [1, 3]);
    column.to_data().assert_approx_eq::<f32>(
        &TensorData::from([[0.231_002_03, 0.436_361_58, 0.720_644_24]]),
        Tolerance::absolute(1e-7),
    );
}

#[test]
fn higher_rank_factors_preserve_independent_batches() {
    let device = Device::default();
    let optimizer = AdafactorConfig::new()
        .with_epsilon_1(0.01)
        .with_clip_threshold(10.0)
        .with_relative_step(false)
        .with_scale_parameter(false)
        .build();
    let grad = Tensor::<3>::from_floats(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[2.0, -1.0, 4.0], [0.5, 3.0, -2.0]],
        ],
        &device,
    );

    let (tensor, state) = optimizer.step(0.2, Tensor::<3>::zeros([2, 2, 3], &device), grad, None);

    tensor.to_data().assert_approx_eq::<f32>(
        &TensorData::from([
            [
                [-0.123_505_19, -0.189_167_4, -0.227_815_61],
                [-0.210_835_67, -0.201_829_75, -0.194_451_99],
            ],
            [
                [-0.247_250_27, 0.080_702_476, -0.228_375_06],
                [-0.077_785_276, -0.304_669_32, 0.143_694_2],
            ],
        ]),
        Tolerance::absolute(1e-6),
    );
    let state = state.unwrap();
    assert_eq!(state.second_moment.dims(), [2, 2, 1]);
    state.second_moment.to_data().assert_approx_eq::<f32>(
        &TensorData::from([[[4.676_666_7], [25.676_666]], [[7.01], [4.426_666_7]]]),
        Tolerance::absolute(3e-6),
    );
    let column = state.column_second_moment.unwrap();
    assert_eq!(column.dims(), [2, 1, 3]);
    column.to_data().assert_approx_eq::<f32>(
        &TensorData::from([[[8.51, 14.51, 22.51]], [[2.135, 5.01, 10.01]]]),
        Tolerance::absolute(3e-6),
    );
}

#[test]
fn relative_step_caps_supplied_learning_rate_and_decays_with_time() {
    let device = Device::default();
    let optimizer = AdafactorConfig::new().with_scale_parameter(false).build();
    let tensor = Tensor::<1>::from_floats([2.0, -2.0], &device);
    let grad = Tensor::<1>::ones([2], &device);

    let (small_step, _) = optimizer.step(0.1, tensor.clone(), grad.clone(), None);
    small_step
        .to_data()
        .assert_approx_eq::<f32>(&TensorData::from([1.9, -2.1]), Tolerance::absolute(1e-6));
    let (tensor, state) = optimizer.step(2.0, tensor, grad.clone(), None);
    tensor
        .to_data()
        .assert_eq(&TensorData::from([1.0f32, -3.0]), true);
    let (tensor, _) = optimizer.step(2.0, tensor, grad, state);
    tensor.to_data().assert_approx_eq::<f32>(
        &TensorData::from([0.292_893_23, -3.707_106_8]),
        Tolerance::absolute(1e-6),
    );
}

#[test]
fn absolute_step_uses_supplied_learning_rate_without_a_cap() {
    let device = Device::default();
    let optimizer = AdafactorConfig::new()
        .with_relative_step(false)
        .with_scale_parameter(false)
        .build();
    let (tensor, _) = optimizer.step(
        2.0,
        Tensor::<1>::from_floats([2.0, -2.0], &device),
        Tensor::<1>::ones([2], &device),
        None,
    );
    tensor
        .to_data()
        .assert_eq(&TensorData::from([0.0f32, -4.0]), true);
}

#[test]
fn parameter_scale_uses_rms_with_an_epsilon_floor() {
    let device = Device::default();
    let optimizer = AdafactorConfig::new().with_epsilon_2(0.1).build();
    let (tensor, _) = optimizer.step(
        0.01,
        Tensor::<1>::from_floats([3.0, 4.0], &device),
        Tensor::<1>::ones([2], &device),
        None,
    );
    tensor.to_data().assert_approx_eq::<f32>(
        &TensorData::from([2.964_644_7, 3.964_644_7]),
        Tolerance::absolute(1e-6),
    );

    let (tensor, _) = optimizer.step(
        0.2,
        Tensor::<1>::zeros([2], &device),
        Tensor::<1>::ones([2], &device),
        None,
    );
    tensor
        .to_data()
        .assert_approx_eq::<f32>(&TensorData::from([-0.02, -0.02]), Tolerance::absolute(1e-7));
}

#[test]
fn update_clipping_limits_update_rms() {
    let device = Device::default();
    let optimizer = AdafactorConfig::new()
        .with_clip_threshold(0.25)
        .with_scale_parameter(false)
        .build();
    let (tensor, _) = optimizer.step(
        0.2,
        Tensor::<1>::from_floats([1.0, -2.0], &device),
        Tensor::<1>::from_floats([1.0, -1.0], &device),
        None,
    );
    tensor
        .to_data()
        .assert_approx_eq::<f32>(&TensorData::from([0.95, -1.95]), Tolerance::absolute(1e-6));
}

#[test]
fn weight_decay_uses_supplied_learning_rate_and_original_parameter() {
    let device = Device::default();
    let optimizer = AdafactorConfig::new().with_weight_decay(0.1).build();
    let (tensor, _) = optimizer.step(
        2.0,
        Tensor::<1>::from_floats([3.0, 4.0], &device),
        Tensor::<1>::ones([2], &device),
        None,
    );
    // The adaptive step is RMS([3, 4]); weight decay is 2 * 0.1 times the parameter.
    tensor.to_data().assert_approx_eq::<f32>(
        &TensorData::from([-1.135_533_9, -0.335_533_92]),
        Tolerance::absolute(1e-6),
    );
}

#[test]
fn zero_gradients_remain_finite_in_both_state_representations() {
    let device = Device::default();
    let optimizer = AdafactorConfig::new().build();
    let (vector, state) = optimizer.step(
        0.1,
        Tensor::<1>::zeros([3], &device),
        Tensor::<1>::zeros([3], &device),
        None,
    );
    let (vector, _) = optimizer.step(0.1, vector, Tensor::<1>::zeros([3], &device), state);
    vector
        .to_data()
        .assert_eq(&TensorData::from([0.0f32, 0.0, 0.0]), true);

    let (matrix, state) = optimizer.step(
        0.1,
        Tensor::<2>::zeros([2, 3], &device),
        Tensor::<2>::zeros([2, 3], &device),
        None,
    );
    let (matrix, _) = optimizer.step(0.1, matrix, Tensor::<2>::zeros([2, 3], &device), state);
    matrix
        .to_data()
        .assert_eq(&TensorData::from([[0.0f32; 3]; 2]), true);
}

#[test]
fn reduced_precision_parameters_keep_dtype_and_f32_state() {
    let device = Device::default();
    let optimizer = AdafactorConfig::new().with_scale_parameter(false).build();

    for dtype in [FloatDType::F16, FloatDType::BF16] {
        if !device.supports_dtype(dtype) {
            continue;
        }
        let tensor = Tensor::<2>::from_floats([[1.0, -2.0], [3.0, -4.0]], &device).cast(dtype);
        let original_dtype = tensor.dtype();
        // With default epsilon, computing the state in f16 would underflow to zero.
        let grad = Tensor::<2>::zeros([2, 2], &device).cast(dtype);
        let (tensor, state) = optimizer.step(0.1, tensor, grad.clone(), None);
        let (tensor, state) = optimizer.step(0.1, tensor, grad, state);

        assert_eq!(tensor.dtype(), original_dtype);
        tensor.to_data().assert_approx_eq::<f32>(
            &TensorData::from([[1.0, -2.0], [3.0, -4.0]]),
            Tolerance::absolute(1e-6),
        );
        let state = state.unwrap();
        assert_eq!(state.second_moment.dtype(), DType::F32);
        assert_eq!(state.column_second_moment.unwrap().dtype(), DType::F32);
    }
}

#[test]
fn module_optimizer_applies_configured_gradient_clipping() {
    let device = Device::default().autodiff();
    let linear = Linear {
        weight: Param::from_data(TensorData::from([[1.0, -2.0], [3.0, -4.0]]), &device),
        bias: Some(Param::from_data(TensorData::from([0.5, -0.5]), &device)),
    };
    let input = Tensor::<2>::from_floats([[0.25, -0.75]], &device).require_grad();
    let mut optimizer = AdafactorConfig::new()
        .with_epsilon_1(1.0)
        .with_relative_step(false)
        .with_scale_parameter(false)
        .with_grad_clipping(Some(GradientClippingConfig::Value(0.1)))
        .init();
    let grads = GradientsParams::from_grads(linear.forward(input).backward(), &linear);

    let linear = optimizer.step(0.1, linear, grads);

    linear.weight.to_data().assert_approx_eq::<f32>(
        &TensorData::from([[0.990_049_6, -2.009_950_4], [3.009_950_4, -3.990_049_6]]),
        Tolerance::absolute(1e-6),
    );
    linear.bias.unwrap().to_data().assert_approx_eq::<f32>(
        &TensorData::from([0.490_049_63, -0.509_950_4]),
        Tolerance::absolute(1e-6),
    );
}

#[test]
fn factored_and_unfactored_state_survive_burnpack_round_trip() {
    let device = Device::default().autodiff();
    let linear = Linear {
        weight: Param::from_data(TensorData::from([[1.0, -2.0], [3.0, -4.0]]), &device),
        bias: Some(Param::from_data(TensorData::from([0.5, -0.5]), &device)),
    };
    let config = AdafactorConfig::new().with_weight_decay(0.1);
    let mut optimizer = config.init();
    let input = Tensor::<2>::from_floats([[0.25, -0.75]], &device).require_grad();
    let grads = GradientsParams::from_grads(linear.forward(input).backward(), &linear);
    let linear = optimizer.step(0.01, linear, grads);
    let bytes = optimizer.into_bytes().unwrap();
    assert!(!bytes.is_empty());
    let mut reloaded = config.init().from_bytes(bytes).unwrap();

    // A different gradient makes restoring the first step's moments observable.
    let input = Tensor::<2>::from_floats([[-0.5, 1.5]], &device).require_grad();
    let grads_original =
        GradientsParams::from_grads(linear.forward(input.clone()).square().backward(), &linear);
    let grads_reloaded =
        GradientsParams::from_grads(linear.forward(input).square().backward(), &linear);
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

#[test]
#[should_panic(expected = "epsilon_1 must be positive and finite")]
fn rejects_zero_squared_gradient_epsilon() {
    AdafactorConfig::new().with_epsilon_1(0.0).build();
}

#[test]
#[should_panic(expected = "epsilon_2 must be positive and finite")]
fn rejects_negative_parameter_scale_epsilon() {
    AdafactorConfig::new().with_epsilon_2(-1.0).build();
}

#[test]
#[should_panic(expected = "clip_threshold must be positive and finite")]
fn rejects_nan_clipping_threshold() {
    AdafactorConfig::new().with_clip_threshold(f32::NAN).build();
}

#[test]
#[should_panic(expected = "decay_rate must be negative and finite")]
fn rejects_nonnegative_decay_exponent() {
    AdafactorConfig::new().with_decay_rate(0.0).build();
}

#[test]
#[should_panic(expected = "weight_decay must be nonnegative and finite")]
fn rejects_infinite_weight_decay() {
    AdafactorConfig::new()
        .with_weight_decay(f32::INFINITY)
        .build();
}
