use crate::*;
use burn_tensor::{TensorData, activation};

#[test]
fn should_diff_relu() {
    let data_1 = TensorData::from([[1.0, 7.0], [-2.0, -3.0]]);
    let data_2 = TensorData::from([[4.0, -7.0], [2.0, 3.0]]);

    let device = Default::default();
    let tensor_1 = TestAutodiffTensor::<2>::from_data(data_1, &device).require_grad();
    let tensor_2 = TestAutodiffTensor::from_data(data_2, &device).require_grad();

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
