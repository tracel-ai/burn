use crate::*;
use burn_tensor::TensorData;

#[test]
fn should_diff_full_complex_1() {
    let data_1 = TensorData::from([[1.0, 7.0], [13.0, -3.0]]);
    let data_2 = TensorData::from([[4.0, 7.0], [2.0, 3.0]]);

    let device = Default::default();
    let tensor_1 = TestAutodiffTensor::<2>::from_data(data_1, &device).require_grad();
    let tensor_2 = TestAutodiffTensor::from_data(data_2, &device).require_grad();

    let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
    let tensor_4 = tensor_3.matmul(tensor_1.clone());
    let tensor_5 = tensor_4.mul(tensor_2.clone());

    let grads = tensor_5.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    grad_1
        .to_data()
        .assert_eq(&TensorData::from([[593., 463.0], [487.0, 539.0]]), false);
    grad_2
        .to_data()
        .assert_eq(&TensorData::from([[734.0, 294.0], [1414.0, 242.0]]), false);
}

#[test]
fn should_diff_full_complex_2() {
    let data_1 = TensorData::from([[1.0, 7.0], [13.0, -3.0]]);
    let data_2 = TensorData::from([[4.0, 7.0], [2.0, 3.0]]);

    let device = Default::default();
    let tensor_1 = TestAutodiffTensor::<2>::from_data(data_1, &device).require_grad();
    let tensor_2 = TestAutodiffTensor::from_data(data_2, &device).require_grad();

    let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
    let tensor_4 = tensor_3.matmul(tensor_1.clone());
    let tensor_5 = tensor_4.add_scalar(17.0).add(tensor_2.clone());

    let grads = tensor_5.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    grad_1
        .to_data()
        .assert_eq(&TensorData::from([[166.0, 110.0], [212.0, 156.0]]), false);
    grad_2
        .to_data()
        .assert_eq(&TensorData::from([[113.0, 141.0], [33.0, 41.0]]), false);
}

#[test]
fn should_diff_full_complex_3() {
    let data_1 = TensorData::from([[1.0, 7.0], [13.0, -3.0]]);
    let data_2 = TensorData::from([[4.0, 7.0], [2.0, 3.0]]);

    let device = Default::default();
    let tensor_1 = TestAutodiffTensor::<2>::from_data(data_1, &device).require_grad();
    let tensor_2 = TestAutodiffTensor::from_data(data_2, &device).require_grad();

    let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
    let tensor_4 = tensor_3.matmul(tensor_1.clone());
    let tensor_5 = tensor_4.clone().sub(tensor_2.clone());
    let tensor_6 = tensor_5.add(tensor_4);

    let grads = tensor_6.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    grad_1
        .to_data()
        .assert_eq(&TensorData::from([[332.0, 220.0], [424.0, 312.0]]), false);
    grad_2
        .to_data()
        .assert_eq(&TensorData::from([[223.0, 279.0], [63.0, 79.0]]), false);
}
