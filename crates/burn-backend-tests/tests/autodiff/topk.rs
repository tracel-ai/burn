use super::*;
use burn_tensor::{TensorData, Tolerance};

#[test]
fn should_diff_topk() {
    let device = AutodiffDevice::new();
    let tensor = TestTensor::<1>::from_data([3.0, 8.0, 2.0, 5.0], &device).require_grad();
    let weights = TestTensor::<1>::from_data([2.0, 3.0], &device);

    let values = tensor.clone().topk(2, 0);
    values
        .clone()
        .into_data()
        .assert_approx_eq::<FloatElem>(&TensorData::from([8.0, 5.0]), Tolerance::default());

    let grads = values.mul(weights).backward();
    let grad = tensor.grad(&grads).unwrap();

    grad.into_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([0.0, 2.0, 0.0, 3.0]),
        Tolerance::default(),
    );
}

#[test]
fn should_diff_topk_along_non_final_dim() {
    let device = AutodiffDevice::new();
    let tensor =
        TestTensor::<3>::from_data([[[1.0, 6.0], [5.0, 2.0], [3.0, 4.0]]], &device).require_grad();
    let weights = TestTensor::<3>::from_data([[[2.0, 3.0], [4.0, 5.0]]], &device);

    let values = tensor.clone().topk(2, 1);
    values.clone().into_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([[[5.0, 6.0], [3.0, 4.0]]]),
        Tolerance::default(),
    );

    let grads = values.mul(weights).backward();
    let grad = tensor.grad(&grads).unwrap();

    grad.into_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([[[0.0, 3.0], [2.0, 0.0], [4.0, 5.0]]]),
        Tolerance::default(),
    );
}

#[test]
fn should_diff_topk_when_k_equals_dimension_size() {
    let device = AutodiffDevice::new();
    let tensor = TestTensor::<2>::from_data([[1.0, 4.0], [3.0, 2.0]], &device).require_grad();
    let weights = TestTensor::<2>::from_data([[2.0, 3.0], [4.0, 5.0]], &device);

    let values = tensor.clone().topk(2, 0);
    let grads = values.mul(weights).backward();
    let grad = tensor.grad(&grads).unwrap();

    grad.into_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([[4.0, 3.0], [2.0, 5.0]]),
        Tolerance::default(),
    );
}
