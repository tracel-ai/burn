use crate::qtensor::*;
use crate::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
fn should_support_remainder_basic() {
    let lhs = QTensor::<TestBackend, 1>::int8([-3.0, -2.0, -1.0, 1.0, 2.0, 2.0]);
    let rhs = QTensor::<TestBackend, 1>::int8([2.0, 3.0, 1.0, 2.0, 1.0, 2.0]);

    let output = lhs.remainder(rhs);
    let expected = TensorData::from([1., 1., 0., 1., 0., 0.]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-1));
}

#[test]
fn should_support_remainder_basic_scalar() {
    let tensor = QTensor::<TestBackend, 1>::int8([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]);

    let output = tensor.remainder_scalar(2.0);
    let expected = TensorData::from([1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-1));
}

#[test]
fn should_support_remainder_float() {
    let lhs = QTensor::<TestBackend, 1>::int8([1.0, 2.0, 3.0, 4.0, 5.0]);
    let rhs = QTensor::<TestBackend, 1>::int8([1.4233, 2.7313, 0.2641, 1.9651, 0.5897]);

    let output = lhs.remainder(rhs);
    let expected = TensorData::from([1., 2., 0.0949, 0.0698, 0.2824]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-1));
}

#[test]
fn should_support_remainder_float_scalar() {
    let tensor = QTensor::<TestBackend, 1>::int8([1.0, 2.0, 3.0, 4.0, 5.0]);

    let output = tensor.remainder_scalar(-1.5);
    let expected = TensorData::from([-0.5, -1.0, 0.0, -0.5, -1.0]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-1));
}

#[test]
fn should_be_zero() {
    let lhs = QTensor::<TestBackend, 1>::int8([0.0, 0.0, 0.0]);
    let rhs = QTensor::<TestBackend, 1>::int8([3.5, -2.1, 1.5]);

    let output = lhs.remainder(rhs);
    let expected = TensorData::from([0.0, 0.0, 0.0]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-1));
}

#[test]
fn should_be_zero_scalar() {
    let tensor = QTensor::<TestBackend, 1>::int8([0.0, 0.0, 0.0]);

    let output = tensor.remainder_scalar(3.5);
    let expected = TensorData::from([0.0, 0.0, 0.0]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-1));
}

#[test]
fn should_have_no_remainder() {
    let lhs = QTensor::<TestBackend, 1>::int8([1.0, 2.0, 3.0, 4.0, 5.0]);
    let rhs = QTensor::<TestBackend, 1>::int8([1.0, 2.0, 3.0, 4.0, 5.0]);

    let output = lhs.remainder(rhs);
    let expected = TensorData::from([0.0, 0.0, 0.0, 0.0, 0.0]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-1));
}

#[test]
fn should_have_no_remainder_scalar() {
    let tensor = QTensor::<TestBackend, 1>::int8([4.0, 4.0]);

    let output = tensor.remainder_scalar(4.0);
    let expected = TensorData::from([0.0, 0.0]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-1));
}

#[test]
fn should_be_negative() {
    let lhs = QTensor::<TestBackend, 1>::int8([-7.0, -3.0, 2.0, 6.0]);
    let rhs = QTensor::<TestBackend, 1>::int8([-2.5, -2.1, -1.5, -3.25]);

    let output = lhs.remainder(rhs);
    let expected = TensorData::from([-2., -0.9, -1., -0.5]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-1));
}

#[test]
fn should_be_negative_scalar() {
    let tensor = QTensor::<TestBackend, 1>::int8([-7.0, -3.0, 2.0, 6.0]);

    let output = tensor.remainder_scalar(-2.5);
    let expected = TensorData::from([-2.0, -0.50, -0.50, -1.5]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-1));
}

#[test]
fn should_support_fp_dividends() {
    let tensor = QTensor::<TestBackend, 1>::int8([-7.5, -2.5, 2.5, 7.5]);

    let output = tensor.remainder_scalar(3.0);
    let expected = TensorData::from([1.5, 0.5, 2.5, 1.5]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-1));
}

#[test]
fn should_support_large_divisor() {
    let lhs = QTensor::<TestBackend, 1>::int8([-1.0, 1.0, -1.5, 1.5, -1.0, 1.0, -1.5, 1.5]);
    let rhs = QTensor::<TestBackend, 1>::int8([10.0, 10.0, 10.0, 10.0, -10.0, -10.0, -10.0, -10.0]);

    let output = lhs.remainder(rhs);
    let expected = TensorData::from([9., 1., 8.5, 1.5, -1., -9., -1.5, -8.5]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-1));
}

#[test]
fn should_support_large_divisor_scalar() {
    let tensor = QTensor::<TestBackend, 1>::int8([-1.0, 1.0]);

    let output = tensor.remainder_scalar(10.0);
    let expected = TensorData::from([9.0, 1.0]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-1));
}

#[test]
fn should_support_remainder_op() {
    let lhs = QTensor::<TestBackend, 1>::int8([-3.0, -2.0, -1.0, 1.0, 2.0, 2.0]);
    let rhs = QTensor::<TestBackend, 1>::int8([2.0, 3.0, 1.0, 2.0, 1.0, 2.0]);

    let output = lhs % rhs;
    let expected = TensorData::from([1., 1., 0., 1., 0., 0.]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-1));
}

#[test]
fn should_support_remainder_scalar_op() {
    let tensor = QTensor::<TestBackend, 1>::int8([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]);

    let output = tensor % 2.0;
    let expected = TensorData::from([1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::absolute(1e-1));
}
