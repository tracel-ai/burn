use crate::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
fn should_diff_permute() {
    let data_1 = TensorData::from([[[1.0, 7.0], [2.0, 3.0]]]); // 1x2x2
    let data_2 = TensorData::from([[[1.0, 7.0], [3.2, 2.0], [3.0, 3.0]]]); // 1x3x2

    let device = Default::default();
    let tensor_1 = TestAutodiffTensor::from_data(data_1, &device).require_grad();
    let tensor_2 = TestAutodiffTensor::from_data(data_2, &device).require_grad();

    let tensor_3 = tensor_2.clone().permute([0, 2, 1]);
    let tensor_4 = tensor_1.clone().matmul(tensor_3);
    let grads = tensor_4.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    let tolerance = Tolerance::default().set_half_precision_relative(1e-3);
    grad_1
        .into_data()
        .assert_approx_eq::<FloatElem>(&TensorData::from([[[7.2, 12.0], [7.2, 12.0]]]), tolerance); // 1x2x2
    grad_2.into_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([[[3.0, 10.0], [3.0, 10.0], [3.0, 10.0]]]),
        tolerance,
    ); // 1x3x2
}
