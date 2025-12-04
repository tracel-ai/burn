use crate::*;
use burn_tensor::{TensorData, Tolerance};

#[test]
fn should_diff_cummin() {
    // Simple test to verify cummin gradients work
    let device = Default::default();
    let tensor = TestAutodiffTensor::<1>::from_data(TensorData::from([3.0, 2.0, 4.0]), &device)
        .require_grad();

    let output = tensor.clone().cummin(0);
    let grads = output.sum().backward();
    let grad = tensor.grad(&grads).unwrap();

    // PyTorch reference: [1.0, 2.0, 0.0]
    let expected = TensorData::from([1.0, 2.0, 0.0]);
    grad.to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_diff_cummin_2d() {
    // Test 2D cummin gradients
    let device = Default::default();
    let tensor = TestAutodiffTensor::<2>::from_data(
        TensorData::from([[3.0, 2.0, 4.0], [5.0, 1.0, 3.0]]),
        &device,
    )
    .require_grad();

    let output = tensor.clone().cummin(1);
    let grads = output.sum().backward();
    let grad = tensor.grad(&grads).unwrap();

    // PyTorch reference: [[1.0, 2.0, 0.0], [1.0, 2.0, 0.0]]
    let expected = TensorData::from([[1.0, 2.0, 0.0], [1.0, 2.0, 0.0]]);
    grad.to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_diff_cummin_duplicate_values() {
    // Test with duplicate minimum values - critical edge case
    let device = Default::default();
    let tensor =
        TestAutodiffTensor::<1>::from_data(TensorData::from([3.0, 2.0, 2.0, 4.0]), &device)
            .require_grad();

    let output = tensor.clone().cummin(0);
    let grads = output.sum().backward();
    let grad = tensor.grad(&grads).unwrap();

    // input:  [3.0, 2.0, 2.0, 4.0]
    // cummin: [3.0, 2.0, 2.0, 2.0]
    // PyTorch reference: [1.0, 1.0, 2.0, 0.0]
    // Position 2 gets grad from itself + position 3
    let expected = TensorData::from([1.0, 1.0, 2.0, 0.0]);
    grad.to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_diff_cummin_all_same() {
    // Test with all same values
    let device = Default::default();
    let tensor = TestAutodiffTensor::<1>::from_data(TensorData::from([2.0, 2.0, 2.0]), &device)
        .require_grad();

    let output = tensor.clone().cummin(0);
    let grads = output.sum().backward();
    let grad = tensor.grad(&grads).unwrap();

    // PyTorch reference: [1.0, 1.0, 1.0]
    // Each position matches cummin, so each gets its own gradient
    let expected = TensorData::from([1.0, 1.0, 1.0]);
    grad.to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_diff_cummin_decreasing() {
    // Test with decreasing sequence
    let device = Default::default();
    let tensor =
        TestAutodiffTensor::<1>::from_data(TensorData::from([5.0, 4.0, 3.0, 2.0]), &device)
            .require_grad();

    let output = tensor.clone().cummin(0);
    let grads = output.sum().backward();
    let grad = tensor.grad(&grads).unwrap();

    // PyTorch reference: [1.0, 1.0, 1.0, 1.0]
    // Each position is a new minimum
    let expected = TensorData::from([1.0, 1.0, 1.0, 1.0]);
    grad.to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_diff_cummin_2d_duplicates() {
    // Test 2D with duplicate values
    let device = Default::default();
    let tensor = TestAutodiffTensor::<2>::from_data(
        TensorData::from([[3.0, 2.0, 2.0, 4.0], [5.0, 1.0, 1.0, 3.0]]),
        &device,
    )
    .require_grad();

    let output = tensor.clone().cummin(1);
    let grads = output.sum().backward();
    let grad = tensor.grad(&grads).unwrap();

    // PyTorch reference: [[1.0, 1.0, 2.0, 0.0], [1.0, 1.0, 2.0, 0.0]]
    let expected = TensorData::from([[1.0, 1.0, 2.0, 0.0], [1.0, 1.0, 2.0, 0.0]]);
    grad.to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}
