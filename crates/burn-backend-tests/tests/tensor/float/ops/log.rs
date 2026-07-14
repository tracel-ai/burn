use super::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;
use core::f32::consts::E;

#[test]
fn should_support_log_3d_transposed() {
    // 3D tensor with permuted dimensions; log should undo the e^k powers.
    let data = TensorData::from([
        [[1.0, E], [E * E, E * E * E]],
        [[1.0, E], [E * E, E * E * E]],
    ]);
    let tensor = TestTensor::<3>::from_data(data, &Default::default());
    let permuted = tensor.permute([2, 0, 1]);

    let output = permuted.log();
    let expected = TensorData::from([[[0.0, 2.0], [0.0, 2.0]], [[1.0, 3.0], [1.0, 3.0]]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_support_log_ops() {
    let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let tensor = TestTensor::<2>::from_data(data, &Default::default());

    let output = tensor.log();
    let expected = TensorData::from([
        [-f32::INFINITY, 0.0, core::f32::consts::LN_2],
        [1.09861, 1.38629, 1.60944],
    ]);

    output.into_data().assert_approx_eq::<FloatElem>(
        &expected,
        Tolerance::default().set_half_precision_relative(1e-3),
    );
}
