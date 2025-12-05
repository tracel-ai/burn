use super::*;
use burn_tensor::{TensorData, Tolerance};

#[test]
fn should_diff_cumsum_dim0() {
    let data_1 = TensorData::from([[1.0, 7.0], [-2.0, -3.0]]);
    let data_2 = TensorData::from([[4.0, -7.0], [2.0, 3.0]]);

    let device = Default::default();
    let tensor_1 = TestAutodiffTensor::<2>::from_data(data_1, &device).require_grad();
    let tensor_2 = TestAutodiffTensor::from_data(data_2, &device).require_grad();

    let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
    let tensor_4 = tensor_3.cumsum(0);
    let tensor_5 = tensor_1.clone().mul(tensor_4);
    let grads = tensor_5.sum().backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    // Expected gradients computed with PyTorch
    let expected = TensorData::from([[-14.0, 24.0], [17.0, 6.0]]);
    grad_1
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());

    let expected = TensorData::from([[3.0, 10.0], [-1.0, 37.0]]);
    grad_2
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_diff_cumsum_dim1() {
    let data_1 = TensorData::from([[1.0, 7.0], [-2.0, -3.0]]);
    let data_2 = TensorData::from([[4.0, -7.0], [2.0, 3.0]]);

    let device = Default::default();
    let tensor_1 = TestAutodiffTensor::<2>::from_data(data_1, &device).require_grad();
    let tensor_2 = TestAutodiffTensor::from_data(data_2, &device).require_grad();

    let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
    let tensor_4 = tensor_3.cumsum(1);
    let tensor_5 = tensor_1.clone().mul(tensor_4);
    let grads = tensor_5.sum().backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    // Expected gradients computed with PyTorch
    let expected = TensorData::from([[1.0, 69.0], [-13.0, -28.0]]);
    grad_1
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());

    let expected = TensorData::from([[18.0, 13.0], [71.0, 58.0]]);
    grad_2
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_diff_cumsum_complex() {
    let data_1 = TensorData::from([[0.0, 1.0], [3.0, 4.0]]);
    let data_2 = TensorData::from([[6.0, 7.0], [9.0, 10.0]]);

    let device = Default::default();
    let tensor_1 = TestAutodiffTensor::<2>::from_data(data_1, &device).require_grad();
    let tensor_2 = TestAutodiffTensor::from_data(data_2, &device).require_grad();

    let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
    let tensor_4 = tensor_3.clone().cumsum(1);
    let tensor_5 = tensor_4.mul(tensor_3);

    let grads = tensor_5.sum().backward();
    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    // Expected gradients computed with PyTorch
    let expected = TensorData::from([[371.0, 542.0], [2246.0, 3281.0]]);
    grad_1
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());

    let expected = TensorData::from([[507.0, 528.0], [704.0, 733.0]]);
    grad_2
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}
