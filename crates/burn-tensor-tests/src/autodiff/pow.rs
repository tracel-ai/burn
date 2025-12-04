use crate::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
fn should_diff_powf_scalar() {
    let data_1 = TensorData::from([[0.0, 1.0], [3.0, 4.0]]);
    let data_2 = TensorData::from([[6.0, 7.0], [9.0, 10.0]]);

    let device = Default::default();
    let tensor_1 = TestAutodiffTensor::<2>::from_data(data_1, &device).require_grad();
    let tensor_2 = TestAutodiffTensor::from_data(data_2, &device).require_grad();

    let tensor_3 = tensor_1.clone().matmul(tensor_2.clone().powf_scalar(0.4));
    let tensor_4 = tensor_3.matmul(tensor_2.clone());
    let grads = tensor_4.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    let tolerance = Tolerance::default().set_half_precision_relative(2e-3);
    let expected = TensorData::from([[68.0, 79.0328], [68.0, 79.0328]]);
    grad_1
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);

    let expected = TensorData::from([[23.5081, 25.2779], [26.0502, 28.6383]]);
    grad_2
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);
}

#[test]
fn should_diff_powf() {
    let device = Default::default();
    let tensor_1 = TestAutodiffTensor::<1>::from_data([2.0, 7.0], &device).require_grad();
    let tensor_2 = TestAutodiffTensor::from_data([4.0, 2.0], &device).require_grad();

    let tensor_3 = tensor_1.clone().powf(tensor_2.clone());
    let grads = tensor_3.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    let expected = TensorData::from([32.0, 14.0]);
    grad_1
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());

    let expected = TensorData::from([11.09035, 95.34960]);
    grad_2
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());

    let expected = TensorData::from([16.0, 49.0]);
    tensor_3
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_diff_powf_with_untracked_lhs() {
    let device = Default::default();
    let tensor_1 = TestAutodiffTensor::<1>::from_data([2.0, 7.0], &device);
    let tensor_2 = TestAutodiffTensor::from_data([4.0, 2.0], &device).require_grad();

    let tensor_3 = tensor_1.clone().powf(tensor_2.clone());
    let grads = tensor_3.backward();

    let grad_2 = tensor_2.grad(&grads).unwrap();

    let expected = TensorData::from([11.09035, 95.34960]);
    grad_2
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_diff_powf_with_untracked_rhs() {
    let device = Default::default();
    let tensor_1 = TestAutodiffTensor::<1>::from_data([2.0, 7.0], &device).require_grad();
    let tensor_2 = TestAutodiffTensor::from_data([4.0, 2.0], &device);

    let tensor_3 = tensor_1.clone().powf(tensor_2.clone());
    let grads = tensor_3.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();

    let expected = TensorData::from([32.0, 14.0]);
    grad_1
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}
