use super::*;
use burn_tensor::TensorData;

#[test]
fn should_diff_ceil() {
    let data = TensorData::from([
        [-1.9751, 0.0714, 0.0643, 0.2406],
        [-1.3172, 0.1252, -0.1119, -0.0127],
    ]);
    let device = Default::default();
    let tensor_1 = TestAutodiffTensor::<2>::from_data(data, &device).require_grad();
    let tensor_2 = tensor_1.clone().ceil();
    let grads = tensor_2.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();

    grad_1.to_data().assert_eq(
        &TensorData::from([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
        false,
    );
}
