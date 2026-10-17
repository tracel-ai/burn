use super::*;
use burn_tensor::{DType, TensorData};

#[test]
fn test_full_precision() {
    let device = AutodiffDevice::new();
    let x1 = TestTensor::<2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device).require_grad();
    let x2 = TestTensor::<2>::from_data([[2.0, 1.0], [1.0, 2.0]], &device).require_grad();
    let dtype = x1.dtype();

    let x3 = x1.clone().cast(DType::F32);
    let x4 = x2.clone().cast(DType::F32);

    let x5 = x3.matmul(x4);
    let x6 = x5.cast(dtype);
    let x7 = x6 * x1.clone() / x2.clone();

    let grads = x7.backward();

    // With M = x1 @ x2 and G = x1 / x2:
    // dx1 = G @ x2^T + M / x2, dx2 = x1^T @ G - M * x1 / x2^2.
    let x1_grad = x1.grad(&grads).unwrap();
    let x2_grad = x2.grad(&grads).unwrap();

    assert_eq!(x1_grad.dtype(), dtype);
    assert_eq!(x2_grad.dtype(), dtype);
    x1_grad
        .into_data()
        .assert_eq(&TensorData::from([[5.0, 9.5], [18.0, 12.5]]), false);
    x2_grad
        .into_data()
        .assert_eq(&TensorData::from([[8.5, -2.0], [-17.0, 1.0]]), false);
}

#[test]
fn one_hot_preserves_autodiff_device() {
    let device = AutodiffDevice::new();
    let targets = TestTensorInt::<1>::from_data([0, 1], &device);
    assert!(targets.device().is_autodiff());
    let one_hot = targets.one_hot::<2>(2).float();

    assert!(one_hot.device().is_autodiff());

    let logits = TestTensor::<2>::ones([2, 2], &device).require_grad();
    let loss = (logits.clone() * one_hot).sum();
    let grads = loss.backward();

    let logits_grad = logits.grad(&grads).expect("logits should have gradients");
    assert!(!logits_grad.device().is_autodiff());
}
