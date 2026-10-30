use super::qtensor::*;
use super::*;
use burn_tensor::{TensorData, Tolerance, ZeroDimPlaceholders};

#[test]
fn should_support_reshape_1d() {
    let tensor = QTensor::<1>::int8([0.0, 1.0, 2.0, 3.0]);

    let output = tensor.reshape([1, 4]);
    let expected = TensorData::from([[0.0, 1.0, 2.0, 3.0]]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(2e-2, 1e-2));
}

#[test]
fn should_support_reshape_2d() {
    let tensor = QTensor::<2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

    let output = tensor.reshape([6]);
    let expected = TensorData::from([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(2e-2, 1e-2));
}

#[test]
fn should_reshape_with_zero_dim_placeholders() {
    let tensor = QTensor::<2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

    let output = tensor.reshape(ZeroDimPlaceholders([0, 1, -1]));
    let expected = TensorData::from([[[0.0, 1.0, 2.0]], [[3.0, 4.0, 5.0]]]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(2e-2, 1e-2));
}

#[test]
fn should_not_corrupt_after_slice() {
    let zeros = QTensor::<1>::int8([0.0, 0.0]);
    zeros.clone().slice([1..2]).reshape([1]).exp();

    // May lead to zeroes being equal to [0.0, 1.0]
    zeros.dequantize().into_data().assert_eq(
        &TestTensor::<1>::zeros([2], &Default::default()).to_data(),
        true,
    );
}
