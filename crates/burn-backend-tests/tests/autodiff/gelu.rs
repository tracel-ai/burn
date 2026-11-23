use super::*;
use burn_tensor::{TensorData, Tolerance, activation};

#[test]
fn should_diff_gelu() {
    let device = AutodiffDevice::new();
    let tensor_1 = TestTensor::<2>::from_data([[0.0, 1.0], [-3.0, 4.0]], &device).require_grad();
    let tensor_2 = TestTensor::from_data([[6.0, -0.5], [9.0, 10.0]], &device).require_grad();

    let x = tensor_1.clone().matmul(activation::gelu(tensor_2.clone()));
    let x = tensor_1.clone().matmul(x);
    let grads = x.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    let tolerance = Tolerance::rel_abs(1e-4, 1e-5)
        .set_half_precision_relative(1e-2)
        .set_half_precision_absolute(1e-2);
    let expected = TensorData::from([[1.46281, 1.46281], [48.22866, 153.46281]]);
    // F32 matmuls may round inputs to TF32; cancellation in this gradient amplifies that error.
    grad_1
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance.set_single_precision_relative(5e-3));

    let expected = TensorData::from([[-15.0000, -1.9875731], [17.0000, 17.0000]]);
    grad_2
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);
}

#[test]
fn should_diff_gelu_elementwise() {
    let device = AutodiffDevice::new();
    let input =
        TestTensor::<1>::from_data([-3.0, -0.5, 0.0, 0.5, 3.0, 6.0], &device).require_grad();
    let weights = TestTensor::<1>::from_data([2.0, -3.0, 4.0, -5.0, 6.0, 7.0], &device);

    let grads = (activation::gelu(input.clone()) * weights).backward();
    let grad = input.grad(&grads).unwrap();

    // weights * (Phi(x) + x * phi(x)), without matmul's optional TF32 rounding.
    let expected = TensorData::from([
        -0.023891294,
        -0.39751463,
        2.0,
        -4.3374756,
        6.071674,
        7.0000002,
    ]);
    let tolerance = Tolerance::rel_abs(1e-4, 1e-5)
        .set_half_precision_relative(1e-2)
        .set_half_precision_absolute(1e-2);
    grad.to_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);
}
