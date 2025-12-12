use super::*;
use burn_tensor::TensorData;

#[test]
fn should_diff_matmul_with_slice() {
    let data_1 = TensorData::from([[1.0, 7.0], [2.0, 3.0]]);
    let data_2 = TensorData::from([[4.0, 7.0, 100.0], [2.0, 3.0, 15.0]]);

    let device = Default::default();
    let tensor_1 = TestAutodiffTensor::<2>::from_data(data_1, &device).require_grad();
    let tensor_2 = TestAutodiffTensor::from_data(data_2, &device).require_grad();

    let tensor_3 = tensor_2.clone().slice([0..2, 0..2]);
    let tensor_4 = tensor_1.clone().matmul(tensor_3);
    let grads = tensor_4.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    grad_1
        .to_data()
        .assert_eq(&TensorData::from([[11.0, 5.0], [11.0, 5.0]]), false);
    grad_2.to_data().assert_eq(
        &TensorData::from([[3.0, 3.0, 0.0], [10.0, 10.0, 0.0]]),
        false,
    );
}

#[test]
fn should_diff_matmul_with_slice_stepped() {
    use burn_tensor::s;
    let data_1 = TensorData::from([[1.0, 7.0], [100.0, 100.0], [2.0, 3.0], [100.0, 100.0]]);
    let data_2 = TensorData::from([[4.0, 100.0, 7.0, 100.0], [2.0, 100.0, 3.0, 15.0]]);

    let device = Default::default();
    let tensor_1 = TestAutodiffTensor::<2>::from_data(data_1, &device).require_grad();
    let tensor_2 = TestAutodiffTensor::from_data(data_2, &device).require_grad();

    let tensor_3 = tensor_1.clone().slice(s![0..;2, 0..2]); // [[1., 7.], [2., 3.]]
    let tensor_4 = tensor_2.clone().slice(s![0..2, 0..;2]); // [[4., 7.], [2., 3.]]
    let tensor_5 = tensor_3.clone().matmul(tensor_4);
    let grads = tensor_5.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    grad_1.to_data().assert_eq(
        &TensorData::from([[11., 5.], [0., 0.], [11., 5.], [0., 0.]]),
        false,
    );
    grad_2.to_data().assert_eq(
        &TensorData::from([[3., 0., 3., 0.], [10., 0., 10., 0.]]),
        false,
    );
}

#[test]
fn should_panic_on_slice_with_step() {
    use burn_tensor::s;

    let data = TensorData::from([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]);
    let device = Default::default();
    let tensor = TestAutodiffTensor::<2>::from_data(data, &device).require_grad();

    // This should panic because step is 2
    let _sliced = tensor.slice(s![.., 0..4; 2]);
}
