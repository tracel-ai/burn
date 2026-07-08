use super::*;
use burn_tensor::{TensorData, Tolerance};

#[test]
fn should_diff_cumprod() {
    // Simple test to verify cumprod gradients work
    let device = AutodiffDevice::new();
    let tensor =
        TestTensor::<1>::from_data(TensorData::from([2.0, 3.0, 4.0]), &device).require_grad();

    let output = tensor.clone().cumprod(0);
    let grads = output.sum().backward();
    let grad = tensor.grad(&grads).unwrap();

    // PyTorch reference: [16.0, 10.0, 6.0]
    let expected = TensorData::from([16.0, 10.0, 6.0]);
    grad.to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_diff_cumprod_2d() {
    // Test 2D cumprod gradients
    let device = AutodiffDevice::new();
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        &device,
    )
    .require_grad();

    let output = tensor.clone().cumprod(1);
    let grads = output.sum().backward();
    let grad = tensor.grad(&grads).unwrap();

    // PyTorch reference: [[9.0, 4.0, 2.0], [36.0, 28.0, 20.0]]
    let expected = TensorData::from([[9.0, 4.0, 2.0], [36.0, 28.0, 20.0]]);
    grad.to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

// The following tests exercise the zero-safe cumprod gradient (issue #3864).
// The gradient is computed as `left[i] * tail[i]` (exclusive prefix product
// times a reverse-accumulated tail) rather than `reverse_cumsum(grad * output)
// / input`, so it stays finite when the input contains zeros. Expected values
// are taken from PyTorch's `torch.cumprod` backward.

#[test]
fn should_diff_cumprod_zero_in_middle() {
    // Test cumprod with zero in the middle - edge case for division
    let device = AutodiffDevice::new();
    let tensor =
        TestTensor::<1>::from_data(TensorData::from([2.0, 0.0, 3.0, 4.0]), &device).require_grad();

    let output = tensor.clone().cumprod(0);
    let grads = output.sum().backward();
    let grad = tensor.grad(&grads).unwrap();

    // PyTorch reference: [1.0, 32.0, 0.0, 0.0]
    let expected = TensorData::from([1.0, 32.0, 0.0, 0.0]);
    grad.to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_diff_cumprod_zero_at_start() {
    // Test cumprod with zero at the beginning
    let device = AutodiffDevice::new();
    let tensor =
        TestTensor::<1>::from_data(TensorData::from([0.0, 2.0, 3.0, 4.0]), &device).require_grad();

    let output = tensor.clone().cumprod(0);
    let grads = output.sum().backward();
    let grad = tensor.grad(&grads).unwrap();

    // PyTorch reference: [33.0, 0.0, 0.0, 0.0]
    let expected = TensorData::from([33.0, 0.0, 0.0, 0.0]);
    grad.to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_diff_cumprod_zero_at_end() {
    // Test cumprod with zero at the end
    let device = AutodiffDevice::new();
    let tensor =
        TestTensor::<1>::from_data(TensorData::from([2.0, 3.0, 4.0, 0.0]), &device).require_grad();

    let output = tensor.clone().cumprod(0);
    let grads = output.sum().backward();
    let grad = tensor.grad(&grads).unwrap();

    // PyTorch reference: [16.0, 10.0, 6.0, 24.0]
    let expected = TensorData::from([16.0, 10.0, 6.0, 24.0]);
    grad.to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_diff_cumprod_multiple_zeros() {
    // Test cumprod with multiple zeros
    let device = AutodiffDevice::new();
    let tensor = TestTensor::<1>::from_data(TensorData::from([2.0, 0.0, 3.0, 0.0, 5.0]), &device)
        .require_grad();

    let output = tensor.clone().cumprod(0);
    let grads = output.sum().backward();
    let grad = tensor.grad(&grads).unwrap();

    // PyTorch reference: [1.0, 8.0, 0.0, 0.0, 0.0]
    let expected = TensorData::from([1.0, 8.0, 0.0, 0.0, 0.0]);
    grad.to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}
