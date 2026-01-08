#![allow(clippy::approx_constant)]

use super::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
fn should_support_cos_ops() {
    let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let tensor = TestTensor::<2>::from_data(data, &Default::default());

    let output = tensor.cos();
    let expected = TensorData::from([[1.0, 0.54030, -0.41615], [-0.98999, -0.65364, 0.28366]]);

    // Metal has less precise trigonometric functions
    let tolerance = Tolerance::default().set_half_precision_relative(1e-2);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);
}

#[test]
fn should_support_cosh_ops() {
    let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let tensor = TestTensor::<2>::from_data(data, &Default::default());

    let output = tensor.cosh();
    let expected = TensorData::from([[1.0000, 1.5431, 3.7622], [10.0677, 27.3082, 74.2099]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_support_sin_ops() {
    let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let tensor = TestTensor::<2>::from_data(data, &Default::default());

    let output = tensor.sin();
    let expected = TensorData::from([[0.0, 0.841471, 0.909297], [0.141120, -0.756802, -0.958924]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_support_sinh_ops() {
    let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let tensor = TestTensor::<2>::from_data(data, &Default::default());

    let output = tensor.sinh();
    let expected = TensorData::from([[0.0000, 1.1752, 3.6269], [10.0179, 27.2899, 74.2032]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_support_tan_ops() {
    let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let tensor = TestTensor::<2>::from_data(data, &Default::default());

    let output = tensor.tan();
    let expected = TensorData::from([[0.0, 1.557408, -2.185040], [-0.142547, 1.157821, -3.380515]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_support_tanh_ops() {
    let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let tensor = TestTensor::<2>::from_data(data, &Default::default());

    let output = tensor.tanh();
    let expected = TensorData::from([[0.0, 0.761594, 0.964028], [0.995055, 0.999329, 0.999909]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_support_asin_ops() {
    let data = TensorData::from([[0.0, 0.5, 0.707107], [-0.5, -0.707107, -1.0]]);
    let tensor = TestTensor::<2>::from_data(data, &Default::default());

    let output = tensor.asin();
    let expected = TensorData::from([[0.0, 0.523599, 0.785398], [-0.523599, -0.785398, -1.570796]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_support_acos_ops() {
    let data = TensorData::from([[0.0, 0.5, 0.707107], [-0.5, -0.707107, -1.0]]);
    let tensor = TestTensor::<2>::from_data(data, &Default::default());

    let output = tensor.acos();
    let expected = TensorData::from([
        [1.570796, 1.047198, 0.785398],
        [2.094395, 2.356194, 3.141593],
    ]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_support_atan_ops() {
    let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let tensor = TestTensor::<2>::from_data(data, &Default::default());

    let output = tensor.atan();
    let expected = TensorData::from([[0.0, 0.785398, 1.107149], [1.249046, 1.325818, 1.373401]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_support_asinh_ops() {
    let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let tensor = TestTensor::<2>::from_data(data, &Default::default());

    let output = tensor.asinh();
    let expected = TensorData::from([[0.0, 0.881374, 1.443635], [1.818446, 2.094713, 2.312438]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_support_acosh_ops() {
    let data = TensorData::from([[1.0, 1.5, 2.0], [3.0, 4.0, 5.0]]);
    let tensor = TestTensor::<2>::from_data(data, &Default::default());

    let output = tensor.acosh();
    let expected = TensorData::from([[0.0, 0.962424, 1.316958], [1.762747, 2.063437, 2.292432]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_support_atanh_ops() {
    let data = TensorData::from([[0.0, 0.5, 0.707107], [-0.5, -0.707107, -0.9]]);
    let tensor = TestTensor::<2>::from_data(data, &Default::default());

    let output = tensor.atanh();
    let expected = TensorData::from([[0.0, 0.549306, 0.881374], [-0.549306, -0.881374, -1.472219]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_support_atan2_ops() {
    let y = TensorData::from([[0.0, 1.0, 1.0], [-1.0, -1.0, 0.0]]);
    let x = TensorData::from([[1.0, 1.0, 0.0], [1.0, 0.0, -1.0]]);

    let y_tensor = TestTensor::<2>::from_data(y, &Default::default());
    let x_tensor = TestTensor::<2>::from_data(x, &Default::default());

    let output = y_tensor.atan2(x_tensor);
    let expected = TensorData::from([[0.0, 0.785398, 1.570796], [-0.785398, -1.570796, 3.141593]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}
