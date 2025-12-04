use crate::*;
use burn_tensor::{TensorData, Tolerance};

#[test]
fn should_diff_div() {
    let data_1 = TensorData::from([1.0, 7.0]);
    let data_2 = TensorData::from([4.0, 7.0]);

    let device = Default::default();
    let tensor_1 = TestAutodiffTensor::<1>::from_data(data_1, &device).require_grad();
    let tensor_2 = TestAutodiffTensor::from_data(data_2, &device).require_grad();

    let tensor_3 = tensor_1.clone().div(tensor_2.clone());
    let grads = tensor_3.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    let expected = TensorData::from([0.25, 0.14285715]);
    grad_1
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());

    let expected = TensorData::from([-0.0625, -0.14285715]);
    grad_2
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_diff_div_scalar() {
    let data = TensorData::from([1.0, 7.0]);

    let tensor = TestAutodiffTensor::<1>::from_data(data, &Default::default()).require_grad();
    let tensor_out = tensor.clone().div_scalar(4.0);

    let grads = tensor_out.backward();
    let grad = tensor.grad(&grads).unwrap();

    grad.to_data()
        .assert_eq(&TensorData::from([0.25, 0.25]), false);
}

#[test]
fn test_div_complex_1() {
    let data_1 = TensorData::from([[1.0, 7.0], [13.0, -3.0]]);
    let data_2 = TensorData::from([[4.0, 7.0], [2.0, 3.0]]);
    let data_3 = TensorData::from([[2.0, 2.0], [2.0, 2.0]]);

    let device = Default::default();
    let tensor_1 = TestAutodiffTensor::<2>::from_data(data_1, &device).require_grad();
    let tensor_2 = TestAutodiffTensor::from_data(data_2, &device).require_grad();
    let tensor_3 = TestAutodiffTensor::from_data(data_3, &device).require_grad();

    let tensor_4 = tensor_1.clone().div(tensor_2.clone());
    let tensor_5 = tensor_4.div(tensor_3.clone());

    let grads = tensor_5.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();
    let grad_3 = tensor_3.grad(&grads).unwrap();

    let expected = TensorData::from([[0.1250, 0.07142857], [0.25, 0.16666667]]);
    grad_1
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());

    let expected = TensorData::from([[-0.03125, -0.07142857], [-1.6250, 0.16666667]]);
    grad_2
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
    let expected = TensorData::from([[-0.0625, -0.25], [-1.6250, 0.25]]);
    grad_3
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn test_div_complex_2() {
    let data_1 = TensorData::from([[0.0, 1.0], [3.0, 4.0]]);
    let data_2 = TensorData::from([[6.0, 7.0], [9.0, 10.0]]);

    let device = Default::default();
    let tensor_1 = TestAutodiffTensor::<2>::from_data(data_1, &device).require_grad();
    let tensor_2 = TestAutodiffTensor::from_data(data_2, &device).require_grad();

    let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
    let tensor_4 = tensor_3.div(tensor_2.clone());

    let grads = tensor_4.backward();
    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    let tolerance = Tolerance::default().set_half_precision_absolute(2e-3);
    let expected = TensorData::from([[2.00, 2.92857146], [1.36666667, 2.0]]);
    grad_1
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);

    let expected = TensorData::from([[0.08333334, 0.09591837], [-0.05555558, -0.06714284]]);
    grad_2
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);
}
