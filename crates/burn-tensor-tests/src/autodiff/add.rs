use crate::*;
use burn_tensor::TensorData;

#[test]
fn should_diff_add() {
    let device = Default::default();
    let tensor_1 = TestAutodiffTensor::<1>::from_floats([2.0, 5.0], &device).require_grad();
    let tensor_2 = TestAutodiffTensor::from_floats([4.0, 1.0], &device).require_grad();

    let tensor_3 = tensor_1.clone() + tensor_2.clone();
    let grads = tensor_3.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    grad_1
        .to_data()
        .assert_eq(&TensorData::from([1.0, 1.0]), false);
    grad_2
        .to_data()
        .assert_eq(&TensorData::from([1.0, 1.0]), false);
    tensor_3
        .to_data()
        .assert_eq(&TensorData::from([6.0, 6.0]), false);
}

#[test]
fn should_diff_add_scalar() {
    let data = TensorData::from([2.0, 10.0]);

    let tensor = TestAutodiffTensor::<1>::from_data(data, &Default::default()).require_grad();
    let tensor_out = tensor.clone().add_scalar(5.0);
    let grads = tensor_out.backward();

    let grad = tensor.grad(&grads).unwrap();

    grad.to_data()
        .assert_eq(&TensorData::from([1.0, 1.0]), false);
    tensor_out
        .into_data()
        .assert_eq(&TensorData::from([7.0, 15.0]), false);
}

#[test]
fn test_add_complex_1() {
    let data_1 = TensorData::from([[1.0, 7.0], [13.0, -3.0]]);
    let data_2 = TensorData::from([[4.0, 7.0], [2.0, 3.0]]);
    let data_3 = TensorData::from([[2.0, 2.0], [2.0, 2.0]]);

    let device = Default::default();
    let tensor_1 = TestAutodiffTensor::<2>::from_data(data_1, &device).require_grad();
    let tensor_2 = TestAutodiffTensor::from_data(data_2, &device).require_grad();
    let tensor_3 = TestAutodiffTensor::from_data(data_3, &device).require_grad();

    let tensor_4 = tensor_1.clone().add(tensor_2.clone());
    let tensor_5 = tensor_4
        .add(tensor_3)
        .add_scalar(5.0)
        .add(tensor_1.clone())
        .add(tensor_2.clone());
    let tensor_6 = tensor_1.clone().add(tensor_5);

    let grads = tensor_6.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    grad_1
        .to_data()
        .assert_eq(&TensorData::from([[3.0, 3.0], [3.0, 3.0]]), false);
    grad_2
        .to_data()
        .assert_eq(&TensorData::from([[2.0, 2.0], [2.0, 2.0]]), false);
}
