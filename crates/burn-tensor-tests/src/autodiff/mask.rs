use super::*;
use burn_tensor::Tolerance;
use burn_tensor::{Bool, Tensor, TensorData};

#[test]
fn should_diff_mask_fill() {
    let data_1 = TensorData::from([[1.0, 7.0], [2.0, 3.0]]);
    let data_2 = TensorData::from([[4.0, 7.0], [2.0, 3.0]]);
    let mask = TensorData::from([[true, false], [false, true]]);

    let device = Default::default();
    let tensor_1 = TestAutodiffTensor::from_data(data_1, &device).require_grad();
    let tensor_2 = TestAutodiffTensor::from_data(data_2, &device).require_grad();
    let mask = Tensor::<TestAutodiffBackend, 2, Bool>::from_bool(mask, &device);

    let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
    let tensor_4 = tensor_3.mask_fill(mask, 2.0);
    let grads = tensor_4.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    grad_1
        .to_data()
        .assert_eq(&TensorData::from([[7.0, 3.0], [4.0, 2.0]]), false);
    grad_2
        .to_data()
        .assert_eq(&TensorData::from([[2.0, 1.0], [3.0, 7.0]]), false);
}

#[test]
fn should_diff_mask_where() {
    let device = Default::default();
    let tensor_1 = TestAutodiffTensor::from_data([[1.0, 7.0], [2.0, 3.0]], &device).require_grad();
    let tensor_2 = TestAutodiffTensor::from_data([[4.0, 7.0], [2.0, 3.0]], &device).require_grad();
    let tensor_3 =
        TestAutodiffTensor::from_data([[8.8, 9.8], [10.8, 11.8]], &device).require_grad();
    let mask =
        Tensor::<TestAutodiffBackend, 2, Bool>::from_data([[true, false], [false, true]], &device);

    let tensor_4 = tensor_1.clone().matmul(tensor_2.clone());
    let tensor_5 = tensor_4.clone().matmul(tensor_3.clone());
    let tensor_6 = tensor_5.mask_where(mask, tensor_3.clone());
    let grads = tensor_6.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();
    let grad_3 = tensor_3.grad(&grads).unwrap();

    let tolerance = Tolerance::default().set_half_precision_relative(1e-3);
    let expected = TensorData::from([[121.8, 55.0], [110.8, 50.0]]);
    grad_1
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);

    let expected = TensorData::from([[27.4, 33.4], [95.0, 115.0]]);
    grad_2
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);

    let expected = TensorData::from([[15., 18.], [23., 29.]]);
    grad_3
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);
}
