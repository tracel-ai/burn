use super::*;
use burn_tensor::{IndexingUpdateOp, TensorData};

#[test]
fn test_select_grad() {
    let device = AutodiffDevice::new();
    let tensor_1 = TestTensor::<2>::from_data(
        TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]),
        &device,
    )
    .require_grad();
    let indices = TestTensorInt::<1>::from_data(TensorData::from([1, 0]), &device);

    let tensor_2 = tensor_1.clone().matmul(tensor_1.clone().transpose());
    let tensor_3 = tensor_1.clone().select(0, indices);
    let tensor_4 = tensor_2.matmul(tensor_3);

    let grads = tensor_4.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();

    grad_1.into_data().assert_eq(
        &TensorData::from([[109., 148., 187.], [37., 58., 79.]]),
        false,
    );
}

#[test]
fn test_select_add_grad() {
    let device = AutodiffDevice::new();
    let tensor_1 = TestTensor::<2>::from_data(
        TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]),
        &device,
    )
    .require_grad();
    let values = TestTensor::from_data(
        TensorData::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        &device,
    )
    .require_grad();
    let indices = TestTensorInt::<1>::from_data(TensorData::from([1, 0]), &device);

    let tensor_2 = tensor_1.clone().matmul(tensor_1.clone().transpose());
    let tensor_3 =
        tensor_1
            .clone()
            .select_assign(0, indices, values.clone(), IndexingUpdateOp::Add);
    let tensor_4 = tensor_2.matmul(tensor_3);

    let grads = tensor_4.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = values.grad(&grads).unwrap();

    grad_1.into_data().assert_eq(
        &TensorData::from([[127., 199., 271.], [172., 244., 316.]]),
        false,
    );
    grad_2
        .into_data()
        .assert_eq(&TensorData::from([[64., 64., 64.], [19., 19., 19.]]), false);
}

#[test]
fn test_select_assign_grad() {
    let device = AutodiffDevice::new();
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]),
        &device,
    )
    .require_grad();
    let values = TestTensor::from_data(TensorData::from([[10.0, 20.0], [30.0, 40.0]]), &device)
        .require_grad();
    let indices = TestTensorInt::<1>::from_data(TensorData::from([3, 1]), &device);

    let result = tensor
        .clone()
        .select_assign(1, indices, values.clone(), IndexingUpdateOp::Assign);

    result.clone().into_data().assert_eq(
        &TensorData::from([[0.0, 20.0, 2.0, 10.0], [4.0, 40.0, 6.0, 30.0]]),
        false,
    );

    let grads = result.sum().backward();
    let grad_tensor = tensor.grad(&grads).unwrap();
    let grad_values = values.grad(&grads).unwrap();

    grad_tensor.into_data().assert_eq(
        &TensorData::from([[1., 0., 1., 0.], [1., 0., 1., 0.]]),
        false,
    );
    grad_values
        .into_data()
        .assert_eq(&TensorData::from([[1., 1.], [1., 1.]]), false);
}

#[test]
fn test_select_assign_mul_grad() {
    let device = AutodiffDevice::new();
    let tensor = TestTensor::<2>::from_data(
        TensorData::from([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]),
        &device,
    )
    .require_grad();
    let values = TestTensor::from_data(TensorData::from([[10.0, 20.0], [30.0, 40.0]]), &device)
        .require_grad();
    let indices = TestTensorInt::<1>::from_data(TensorData::from([2, 0]), &device);
    let weights = TestTensor::from_data(
        TensorData::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        &device,
    );

    let result = tensor
        .clone()
        .select_assign(1, indices, values.clone(), IndexingUpdateOp::Mul);

    result.clone().into_data().assert_eq(
        &TensorData::from([[40.0, 3.0, 40.0], [200.0, 6.0, 210.0]]),
        false,
    );

    let grads = result.mul(weights).sum().backward();
    let grad_tensor = tensor.grad(&grads).unwrap();
    let grad_values = values.grad(&grads).unwrap();

    grad_tensor.into_data().assert_eq(
        &TensorData::from([[20.0, 2.0, 30.0], [160.0, 5.0, 180.0]]),
        false,
    );
    grad_values
        .into_data()
        .assert_eq(&TensorData::from([[12.0, 2.0], [42.0, 20.0]]), false);
}

#[test]
fn test_select_add_grad_different_shapes() {
    let device = AutodiffDevice::new();

    let indices = TestTensorInt::from_ints([1], &device);
    let x = TestTensor::<2>::ones([1, 1], &device).require_grad();
    let y = TestTensor::ones([2, 1], &device).require_grad();

    let w = y
        .clone()
        .select_assign(0, indices, x.clone(), IndexingUpdateOp::Add);
    let w = w.matmul(y.clone().transpose());

    let grads = w.backward();
    let x_grad = x.grad(&grads).unwrap();
    let y_grad = y.grad(&grads).unwrap();

    x_grad
        .into_data()
        .assert_eq(&TensorData::from([[2.0]]), false);
    y_grad
        .into_data()
        .assert_eq(&TensorData::from([[5.0], [5.0]]), false);
}
