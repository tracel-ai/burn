use super::*;
use burn_tensor::{IndexingUpdateOp, TensorData};

#[test]
fn test_gather_grad() {
    let device = AutodiffDevice::new();
    let tensor_1 = TestTensor::from_data(
        TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]),
        &device,
    )
    .require_grad();
    let indices = TestTensorInt::<2>::from_data(
        TensorData::from([[2, 1, 0, 1, 2], [1, 0, 2, 1, 0]]),
        &device,
    );

    let tensor_2 = tensor_1.clone().matmul(tensor_1.clone().transpose());
    let tensor_3 = tensor_1.clone().gather(1, indices);
    let tensor_4 = tensor_2.matmul(tensor_3);

    let grads = tensor_4.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();

    grad_1.to_data().assert_eq(
        &TensorData::from([[94., 150., 187.], [242., 305., 304.]]),
        false,
    );
}

#[test]
fn test_scatter_grad() {
    let device = AutodiffDevice::new();
    let tensor_1 = TestTensor::from_data(
        TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]),
        &device,
    )
    .require_grad();
    let values = TestTensor::from_data(
        TensorData::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        &device,
    )
    .require_grad();
    let indices = TestTensorInt::<2>::from_data(TensorData::from([[2, 1, 0], [2, 0, 1]]), &device);

    let tensor_2 = tensor_1.clone().matmul(tensor_1.clone().transpose());
    let tensor_3 = tensor_1
        .clone()
        .scatter(1, indices, values.clone(), IndexingUpdateOp::Add);
    let tensor_4 = tensor_2.matmul(tensor_3);

    let grads = tensor_4.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = values.grad(&grads).unwrap();

    grad_1.to_data().assert_eq(
        &TensorData::from([[127., 181., 235.], [226., 316., 406.]]),
        false,
    );
    grad_2
        .to_data()
        .assert_eq(&TensorData::from([[19., 19., 19.], [64., 64., 64.]]), false);
}

#[cfg(feature = "ndarray")]
#[test]
fn test_scatter_assign_grad() {
    let device = AutodiffDevice::new();
    let tensor = TestTensor::from_data(
        TensorData::from([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]),
        &device,
    )
    .require_grad();
    let values = TestTensor::from_data(TensorData::from([[10.0, 20.0], [30.0, 40.0]]), &device)
        .require_grad();
    let indices = TestTensorInt::<2>::from_data(TensorData::from([[2, 0], [3, 1]]), &device);

    let result = tensor
        .clone()
        .scatter(1, indices, values.clone(), IndexingUpdateOp::Assign);

    result.clone().into_data().assert_eq(
        &TensorData::from([[20.0, 1.0, 10.0, 3.0], [4.0, 40.0, 6.0, 30.0]]),
        false,
    );

    let grads = result.sum().backward();
    let grad_tensor = tensor.grad(&grads).unwrap();
    let grad_values = values.grad(&grads).unwrap();

    grad_tensor.to_data().assert_eq(
        &TensorData::from([[0., 1., 0., 1.], [1., 0., 1., 0.]]),
        false,
    );
    grad_values
        .to_data()
        .assert_eq(&TensorData::from([[1., 1.], [1., 1.]]), false);
}

#[cfg(feature = "ndarray")]
#[test]
fn test_scatter_mul_grad() {
    let device = AutodiffDevice::new();
    let tensor = TestTensor::from_data(
        TensorData::from([[2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0]]),
        &device,
    )
    .require_grad();
    let values = TestTensor::from_data(TensorData::from([[10.0, 20.0], [30.0, 40.0]]), &device)
        .require_grad();
    let indices = TestTensorInt::<2>::from_data(TensorData::from([[3, 0], [1, 2]]), &device);
    let weights = TestTensor::from_data(
        TensorData::from([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]),
        &device,
    );

    let result = tensor
        .clone()
        .scatter(1, indices, values.clone(), IndexingUpdateOp::Mul);

    result.clone().into_data().assert_eq(
        &TensorData::from([[40.0, 3.0, 4.0, 50.0], [6.0, 210.0, 320.0, 9.0]]),
        false,
    );

    let grads = result.mul(weights).sum().backward();
    let grad_tensor = tensor.grad(&grads).unwrap();
    let grad_values = values.grad(&grads).unwrap();

    grad_tensor.to_data().assert_eq(
        &TensorData::from([[20.0, 2.0, 3.0, 40.0], [5.0, 180.0, 280.0, 8.0]]),
        false,
    );
    grad_values
        .to_data()
        .assert_eq(&TensorData::from([[20.0, 2.0], [42.0, 56.0]]), false);
}

#[test]
fn test_scatter_add_grad_partial_indices() {
    let device = AutodiffDevice::new();
    let tensor_1 =
        TestTensor::from_data(TensorData::from([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]), &device)
            .require_grad();
    let tensor_2 =
        TestTensor::from_data(TensorData::from([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]), &device)
            .require_grad();
    let values = TestTensor::from_data(TensorData::from([[4.0, 5.0, 6.0]]), &device).require_grad();
    let indices = TestTensorInt::<2>::from_data(TensorData::from([[2, 1, 0]]), &device);

    let tensor_3 = tensor_1.clone().mul(tensor_2);
    let tensor_4 = tensor_3
        .clone()
        .scatter(1, indices, values.clone(), IndexingUpdateOp::Add);

    let grads = tensor_4.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = values.grad(&grads).unwrap();

    grad_1
        .to_data()
        .assert_eq(&TensorData::from([[1., 2., 3., 4., 5., 6.]]), false);
    grad_2
        .to_data()
        .assert_eq(&TensorData::from([[1., 1., 1.]]), false);
}
