use crate::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
fn should_diff_transpose() {
    let data_1 = TensorData::from([[1.0, 7.0], [2.0, 3.0]]);
    let data_2 = TensorData::from([[4.0, 7.0], [2.0, 3.0]]);

    let device = Default::default();
    let tensor_1 = TestAutodiffTensor::<2>::from_data(data_1, &device).require_grad();
    let tensor_2 = TestAutodiffTensor::from_data(data_2, &device).require_grad();

    let tensor_3 = tensor_1.clone().matmul(tensor_2.clone().transpose());
    let tensor_4 = tensor_3.transpose();
    let grads = tensor_4.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    grad_1.to_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([[6.0, 10.0], [6.0, 10.0]]),
        Tolerance::default(),
    );
    grad_2.to_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([[3.0, 10.0], [3.0, 10.0]]),
        Tolerance::default(),
    );
}

#[test]
fn should_diff_swap_dims() {
    let device = Default::default();
    let tensor_1 = TestAutodiffTensor::<3>::from_floats(
        [[[0.0, 1.0], [3.0, 4.0]], [[6.0, 7.0], [9.0, 10.0]]],
        &device,
    )
    .require_grad();
    let tensor_2 = TestAutodiffTensor::from_floats(
        [[[1.0, 4.0], [2.0, 5.0]], [[7.0, 10.0], [8.0, 11.0]]],
        &device,
    )
    .require_grad();

    let tensor_3 = tensor_1.clone().matmul(tensor_2.clone().swap_dims(0, 2));
    let tensor_4 = tensor_3.matmul(tensor_2.clone().swap_dims(1, 2));
    let grads = tensor_4.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    grad_1.to_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([[[66., 78.], [66., 78.]], [[270., 306.], [270., 306.]]]),
        Tolerance::default(),
    );
    grad_2.to_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([[[22., 286.], [28., 316.]], [[172., 652.], [190., 694.]]]),
        Tolerance::default(),
    );
}
