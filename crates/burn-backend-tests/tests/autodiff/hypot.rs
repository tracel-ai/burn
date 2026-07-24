use super::*;
use burn_tensor::{TensorData, Tolerance};

#[test]
fn should_diff_hypot() {
    let data_a = TensorData::from([[3.0, 4.0], [5.0, 12.0]]);
    let data_b = TensorData::from([[4.0, 3.0], [12.0, 5.0]]);

    let device = AutodiffDevice::new();
    let tensor_a = TestTensor::<2>::from_data(data_a, &device).require_grad();
    let tensor_b = TestTensor::<2>::from_data(data_b, &device).require_grad();

    let tensor_c = tensor_a.clone().hypot(tensor_b.clone());
    let grads = tensor_c.backward();

    let grad_a = tensor_a.grad(&grads).unwrap();
    let grad_b = tensor_b.grad(&grads).unwrap();

    grad_a.to_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([[0.6, 0.8], [0.3846, 0.9231]]),
        Tolerance::default(),
    );
    grad_b.to_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([[0.8, 0.6], [0.9231, 0.3846]]),
        Tolerance::default(),
    );
}
