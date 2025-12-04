use crate::*;
use burn_tensor::{TensorData, Tolerance, activation};

#[test]
fn should_diff_gelu() {
    let device = Default::default();
    let tensor_1 =
        TestAutodiffTensor::<2>::from_floats([[0.0, 1.0], [-3.0, 4.0]], &device).require_grad();
    let tensor_2 =
        TestAutodiffTensor::from_floats([[6.0, -0.5], [9.0, 10.0]], &device).require_grad();

    let x = tensor_1.clone().matmul(activation::gelu(tensor_2.clone()));
    let x = tensor_1.clone().matmul(x);
    let grads = x.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    let tolerance = Tolerance::permissive();
    let expected = TensorData::from([[1.46281, 1.46281], [48.22866, 153.46280]]);
    grad_1
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);

    let expected = TensorData::from([[-15.0000, -1.98757], [17.0000, 17.0000]]);
    grad_2
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);
}
