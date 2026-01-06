use super::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
fn should_diff_log1p() {
    let tensor_1 = TestAutodiffTensor::<2>::from([[0.0, 1.0], [3.0, 4.0]]).require_grad();
    let tensor_2 = TestAutodiffTensor::from([[6.0, 7.0], [9.0, 10.0]]).require_grad();

    let tensor_3 = tensor_1.clone().matmul(tensor_2.clone().log1p());
    let tensor_4 = tensor_3.matmul(tensor_2.clone());
    let grads = tensor_4.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    let tolerance = Tolerance::default().set_half_precision_relative(1e-3);
    let expected = TensorData::from([[64.80622101, 75.49362183], [64.80622101, 75.49362183]]);
    grad_1
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);

    let expected = TensorData::from([[22.92208481, 24.47565651], [24.72780228, 26.86416626]]);

    grad_2
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);
}
