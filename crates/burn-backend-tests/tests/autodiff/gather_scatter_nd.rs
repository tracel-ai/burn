use super::*;
use burn_tensor::TensorData;

#[test]
fn test_scatter_nd_add_grad() {
    let device = AutodiffDevice::new();
    let data: TestTensor<2> = TestTensor::from_data(
        TensorData::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        &device,
    )
    .require_grad();
    let values: TestTensor<2> =
        TestTensor::from_data(TensorData::from([[10.0, 20.0, 30.0]]), &device).require_grad();
    let indices = TestTensorInt::<2>::from_data(TensorData::from([[1]]), &device);

    // scatter_nd_add: data[1, :] += values[0, :]
    let result: TestTensor<2> = data.clone().scatter_nd_add(indices, values.clone());
    let grads = result.sum().backward();

    let grad_data = data.grad(&grads).unwrap();
    let grad_values = values.grad(&grads).unwrap();

    // grad_data = all ones (identity for add)
    grad_data.to_data().assert_eq(
        &TensorData::from([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
        false,
    );
    // grad_values = gather_nd(ones, indices) = ones at row 1
    grad_values
        .to_data()
        .assert_eq(&TensorData::from([[1.0, 1.0, 1.0]]), false);
}

#[test]
fn test_scatter_nd_assign_grad() {
    let device = AutodiffDevice::new();
    let data: TestTensor<2> = TestTensor::from_data(
        TensorData::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        &device,
    )
    .require_grad();
    let values: TestTensor<2> =
        TestTensor::from_data(TensorData::from([[10.0, 20.0, 30.0]]), &device).require_grad();
    let indices = TestTensorInt::<2>::from_data(TensorData::from([[1]]), &device);

    // scatter_nd (assign): data[1, :] = values[0, :]
    let result: TestTensor<2> = data.clone().scatter_nd(indices, values.clone());
    let grads = result.sum().backward();

    let grad_data = data.grad(&grads).unwrap();
    let grad_values = values.grad(&grads).unwrap();

    // grad_data: all ones except row 1 is zeroed out (overwritten positions get no gradient)
    grad_data.to_data().assert_eq(
        &TensorData::from([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        false,
    );
    // grad_values = gather_nd(ones, indices) = ones at row 1
    grad_values
        .to_data()
        .assert_eq(&TensorData::from([[1.0, 1.0, 1.0]]), false);
}

#[test]
fn test_gather_nd_grad() {
    let device = AutodiffDevice::new();
    let data: TestTensor<2> = TestTensor::from_data(
        TensorData::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        &device,
    )
    .require_grad();
    let indices = TestTensorInt::<2>::from_data(TensorData::from([[0], [2]]), &device);

    // gather_nd: extract rows 0 and 2
    let result: TestTensor<2> = data.clone().gather_nd(indices);
    let grads = result.sum().backward();

    let grad_data = data.grad(&grads).unwrap();

    // grad_data: scatter_nd(zeros, indices, ones, Add) -> row 0 and 2 get 1s, row 1 stays 0
    grad_data.to_data().assert_eq(
        &TensorData::from([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        false,
    );
}
