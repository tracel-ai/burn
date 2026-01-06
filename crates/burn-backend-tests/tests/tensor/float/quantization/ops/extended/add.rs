use super::qtensor::*;
use super::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
fn test_add_d2() {
    let tensor_1 = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let tensor_2 = QTensor::<TestBackend, 2>::int8([[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]);

    let output = tensor_1 + tensor_2;

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(
            &TensorData::from([[6.0, 8.0, 10.0], [12.0, 14.0, 16.0]]),
            Tolerance::absolute(1e-1),
        );
}

#[test]
fn test_add_broadcast() {
    let tensor_1 = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0]]);
    let tensor_2 = QTensor::<TestBackend, 2>::int8([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]);

    let output = tensor_1 + tensor_2;

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(
            &TensorData::from([[3.0, 5.0, 7.0], [6.0, 8.0, 10.0]]),
            Tolerance::absolute(1e-1),
        );
}

#[test]
fn test_add_different_strides_rhs() {
    // We need to execute an operation after `from data` to trigger inplace in some backends.
    // Which is the operation that might be problematic in this case.
    let tensor_1 = QTensor::<TestBackend, 2>::int8([[0.0, 1.0], [2.0, 3.0]]) * 1;
    let tensor_2 = QTensor::<TestBackend, 2>::int8([[4.0, 5.0], [6.0, 7.0]]) * 1;

    let output = tensor_1 + tensor_2.transpose();

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(
            &TensorData::from([[4.0, 7.0], [7.0, 10.0]]),
            Tolerance::absolute(1e-1),
        );
}

#[test]
fn test_add_different_strides_lhs() {
    // We need to execute an operation after `from data` to trigger inplace in some backends.
    // Which is the operation that might be problematic in this case.
    let tensor_1 = QTensor::<TestBackend, 2>::int8([[0.0, 1.0], [2.0, 3.0]]) * 1;
    let tensor_2 = QTensor::<TestBackend, 2>::int8([[4.0, 5.0], [6.0, 7.0]]) * 1;

    let output = tensor_1.transpose() + tensor_2;

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(
            &TensorData::from([[4.0, 7.0], [7.0, 10.0]]),
            Tolerance::absolute(1e-1),
        );
}

#[test]
fn test_add_different_strides_broadcast() {
    // We need to execute an operation after `from data` to trigger inplace in some backends.
    // Which is the operation that might be problematic in this case.
    let tensor_1 = QTensor::<TestBackend, 2>::int8([[0.0, 1.0], [2.0, 3.0]]) * 1;
    let tensor_2 = QTensor::<TestBackend, 2>::int8([[4.0, 5.0]]) * 1;

    let output = tensor_1.transpose() + tensor_2;

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(
            &TensorData::from([[4.0, 7.0], [5.0, 8.0]]),
            Tolerance::absolute(1e-1),
        );
}

#[test]
fn should_support_add_scalar_ops() {
    let scalar = 2.0;
    let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

    let output = tensor + scalar;

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(
            &TensorData::from([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]),
            Tolerance::absolute(1e-1),
        );
}
