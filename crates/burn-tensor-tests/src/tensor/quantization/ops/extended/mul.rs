use super::*;
use crate::qtensor::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
fn should_support_mul_ops() {
    let tensor_1 = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let tensor_2 = tensor_1.clone();

    let output = tensor_1 * tensor_2;
    let expected = TensorData::from([[0.0, 1.0, 4.0], [9.0, 16.0, 25.0]]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(5e-2, 1e-2));
}

#[test]
fn test_mul_broadcast() {
    let tensor_1 = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0]]);
    let tensor_2 = QTensor::<TestBackend, 2>::int8([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]);

    let output = tensor_1 * tensor_2;
    let expected = TensorData::from([[0.0, 4.0, 10.0], [0.0, 7.0, 16.0]]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(2e-2, 1e-2));
}

#[test]
fn test_mul_broadcast_2_dims() {
    let tensor_1 = QTensor::<TestBackend, 2>::int8([[0.0], [1.0], [2.0]]);
    let tensor_2 = QTensor::<TestBackend, 2>::int8([[3.0, 4.0, 5.0]]);

    let output = tensor_1 * tensor_2;
    let expected = TensorData::from([[0.0, 0.0, 0.0], [3.0, 4.0, 5.0], [6.0, 8.0, 10.0]]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(2e-2, 1e-2));
}

#[test]
fn should_support_mul_scalar_ops() {
    let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let scalar = 2.0;

    let output = tensor * scalar;
    let expected = TensorData::from([[0.0, 2.0, 4.0], [6.0, 8.0, 10.0]]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(2e-2, 1e-2));
}
