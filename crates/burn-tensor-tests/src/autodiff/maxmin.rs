use crate::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
fn should_diff_max_dim() {
    let device = Default::default();
    let tensor_1 =
        TestAutodiffTensor::<2>::from_floats([[1.0, 7.0], [-2.0, -3.0]], &device).require_grad();
    let tensor_2 =
        TestAutodiffTensor::from_floats([[4.0, -7.0], [2.0, 3.0]], &device).require_grad();

    let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
    let tensor_4 = tensor_1.clone().mul(tensor_3.max_dim(1).unsqueeze());
    let grads = tensor_4.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    let expected = TensorData::from([[50.0, 34.0], [40.0, -10.0]]);
    grad_1
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());

    let expected = TensorData::from([[8.0, 10.0], [56.0, 15.0]]);
    grad_2
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_diff_min_dim() {
    let device = Default::default();
    let tensor_1 =
        TestAutodiffTensor::<2>::from_floats([[1.0, 7.0], [-2.0, -3.0]], &device).require_grad();
    let tensor_2 =
        TestAutodiffTensor::from_floats([[4.0, -7.0], [2.0, 3.0]], &device).require_grad();

    let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
    let tensor_4 = tensor_1.clone().mul(tensor_3.min_dim(1).unsqueeze());
    let grads = tensor_4.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    let expected = TensorData::from([[-42.0, 38.0], [-34.0, -24.0]]);
    grad_1
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());

    let expected = TensorData::from([[10.0, 8.0], [15.0, 56.0]]);
    grad_2
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_diff_min_dim_3d_dim1() {
    let device = Default::default();
    let tensor_1 =
        TestAutodiffTensor::<3>::from_floats([[[1.0, 7.0], [-2.0, -3.0]]], &device).require_grad();
    let tensor_2 =
        TestAutodiffTensor::<3>::from_floats([[[4., -7.], [2., 3.]]], &device).require_grad();

    let tensor_3 = tensor_1.clone().mul(tensor_2.clone());
    let tensor_4 = tensor_3.min_dim(1);

    let grads = tensor_4.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    let expected = TensorData::from([[[0., -7.], [2., 0.]]]);
    grad_1
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());

    let expected = TensorData::from([[[0., 7.], [-2., -0.]]]);
    grad_2
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}
