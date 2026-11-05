use super::*;
use burn_tensor::{TensorData, Tolerance};

#[test]
fn should_diff_fmod() {
    let device = AutodiffDevice::new();
    let tensor_1 =
        TestTensor::<1>::from_data(TensorData::from([5.3, -5.3, 5.3, -5.3, 7.5, 0.5]), &device)
            .require_grad();
    let tensor_2 =
        TestTensor::<1>::from_data(TensorData::from([2.0, 2.0, -2.0, -2.0, 3.0, 4.0]), &device)
            .require_grad();

    let tensor_3 = tensor_1.clone().fmod(tensor_2.clone());
    let grads = tensor_3.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    grad_1.to_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        Tolerance::default(),
    );
    // -trunc(x / y), which differs from remainder's -floor(x / y) when x / y < 0
    grad_2.to_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([-2.0, 2.0, 2.0, -2.0, -2.0, -0.0]),
        Tolerance::default(),
    );
}

#[test]
fn should_diff_fmod_scalar() {
    let device = AutodiffDevice::new();
    let tensor_1 =
        TestTensor::<1>::from_data(TensorData::from([5.3, -5.3, 7.5]), &device).require_grad();

    let tensor_2 = tensor_1.clone().fmod_scalar(2.0);
    let grads = tensor_2.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();

    grad_1
        .to_data()
        .assert_approx_eq::<FloatElem>(&TensorData::from([1.0, 1.0, 1.0]), Tolerance::default());
}
