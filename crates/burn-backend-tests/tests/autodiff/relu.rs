use super::*;
use burn_tensor::{TensorData, activation};

#[test]
fn should_diff_relu() {
    let data_1 = TensorData::from([[1.0, 7.0], [-2.0, -3.0]]);
    let data_2 = TensorData::from([[4.0, -7.0], [2.0, 3.0]]);

    let device = AutodiffDevice::new();
    let tensor_1 = TestTensor::<2>::from_data(data_1, &device).require_grad();
    let tensor_2 = TestTensor::from_data(data_2, &device).require_grad();

    let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
    let tensor_4 = activation::relu(tensor_3);
    let tensor_5 = tensor_4.matmul(tensor_2.clone());
    let grads = tensor_5.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    grad_1
        .to_data()
        .assert_eq(&TensorData::from([[-47.0, 9.0], [-35.0, 15.0]]), false);
    grad_2
        .to_data()
        .assert_eq(&TensorData::from([[15.0, 13.0], [-2.0, 39.0]]), false);
}

// A NaN output must not have its gradient zeroed: the trait default masks on
// `output <= 0`, which is false for NaN. See issue #5609.
#[test]
fn should_keep_grad_for_nan_relu_output() {
    let device = AutodiffDevice::new();
    let tensor =
        TestTensor::<1>::from_data(TensorData::from([f32::NAN, -1.0, 2.0]), &device).require_grad();

    let grads = activation::relu(tensor.clone()).sum().backward();

    tensor
        .grad(&grads)
        .unwrap()
        .to_data()
        .assert_eq(&TensorData::from([1.0, 0.0, 1.0]), false);
}
