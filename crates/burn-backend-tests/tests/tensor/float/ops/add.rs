use super::*;
use burn_tensor::{TensorData, backend::Backend};

#[test]
fn test_add_d2() {
    let tensor_1 = TestTensor::<2>::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let tensor_2 = TestTensor::from([[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]);

    let output = tensor_1 + tensor_2;

    output.into_data().assert_eq(
        &TensorData::from([[6.0, 8.0, 10.0], [12.0, 14.0, 16.0]]),
        false,
    );
}

#[test]
fn test_add_broadcast() {
    let tensor_1 = TestTensor::<2>::from([[0.0, 1.0, 2.0]]);
    let tensor_2 = TestTensor::from([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]);

    let output = tensor_1 + tensor_2;

    output.into_data().assert_eq(
        &TensorData::from([[3.0, 5.0, 7.0], [6.0, 8.0, 10.0]]),
        false,
    );
}

#[test]
fn test_add_different_strides_rhs() {
    // We need to execute an operation after `from data` to trigger inplace in some backends.
    // Which is the operation that might be problematic in this case.
    let tensor_1 = TestTensor::<2>::from([[0.0, 1.0], [2.0, 3.0]]) * 1;
    let tensor_2 = TestTensor::from([[4.0, 5.0], [6.0, 7.0]]) * 1;

    let output = tensor_1 + tensor_2.transpose();

    output
        .into_data()
        .assert_eq(&TensorData::from([[4.0, 7.0], [7.0, 10.0]]), false);
}

#[test]
fn test_add_different_strides_lhs() {
    // We need to execute an operation after `from data` to trigger inplace in some backends.
    // Which is the operation that might be problematic in this case.
    let tensor_1 = TestTensor::<2>::from([[0.0, 1.0], [2.0, 3.0]]) * 1;
    let tensor_2 = TestTensor::from([[4.0, 5.0], [6.0, 7.0]]) * 1;

    let output = tensor_1.transpose() + tensor_2;

    output
        .into_data()
        .assert_eq(&TensorData::from([[4.0, 7.0], [7.0, 10.0]]), false);
}

#[test]
fn test_add_different_strides_broadcast() {
    // We need to execute an operation after `from data` to trigger inplace in some backends.
    // Which is the operation that might be problematic in this case.
    let tensor_1 = TestTensor::<2>::from([[0.0, 1.0], [2.0, 3.0]]) * 1;
    let tensor_2 = TestTensor::from([[4.0, 5.0]]) * 1;

    let output = tensor_1.transpose() + tensor_2;

    output
        .into_data()
        .assert_eq(&TensorData::from([[4.0, 7.0], [5.0, 8.0]]), false);
}

#[test]
fn should_support_add_scalar_ops() {
    let scalar = 2.0;
    let tensor = TestTensor::<2>::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

    let output = tensor + scalar;

    output
        .into_data()
        .assert_eq(&TensorData::from([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]), false);
}

#[test]
fn add_maybe_fused_not_contiguous() {
    let tensor1 = TestTensorInt::arange(0..8, &Default::default()).float();
    let tensor2 = TestTensorInt::arange(8..16, &Default::default()).float();
    let tensor1 = tensor1.reshape([2, 4]);
    let tensor2 = tensor2.reshape([4, 2]);
    let tensor2 = tensor2.swap_dims(0, 1);

    TestBackend::sync(&tensor2.device()).unwrap();

    let output = tensor1 + tensor2;

    output.into_data().assert_eq(
        &TensorData::from([[8.0, 11.0, 14.0, 17.0], [13.0, 16.0, 19.0, 22.0]]),
        false,
    );
}

#[test]
fn add_maybe_fused_not_contiguous_broadcasted() {
    let tensor1 = TestTensorInt::arange(0..8, &Default::default()).float();
    let tensor2 = TestTensorInt::arange(8..10, &Default::default()).float();
    let tensor1 = tensor1.reshape([2, 4]);
    let tensor2 = tensor2.reshape([1, 2]);
    let tensor2 = tensor2.swap_dims(0, 1);

    TestBackend::sync(&tensor2.device()).unwrap();

    let output = tensor2 + tensor1;

    output.into_data().assert_eq(
        &TensorData::from([[8.0, 9.0, 10.0, 11.0], [13.0, 14.0, 15.0, 16.0]]),
        false,
    );
}
