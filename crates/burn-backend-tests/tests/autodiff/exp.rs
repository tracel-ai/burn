use super::*;
use burn_tensor::{TensorData, Tolerance};

#[test]
fn should_diff_exp() {
    let data_1 = TensorData::from([[1.0, 7.0], [-2.0, -3.0]]);
    let data_2 = TensorData::from([[4.0, -7.0], [2.0, 3.0]]);

    let device = Default::default();
    let tensor_1 = TestAutodiffTensor::<2>::from_data(data_1, &device).require_grad();
    let tensor_2 = TestAutodiffTensor::from_data(data_2, &device).require_grad();

    let tensor_3 = tensor_1.clone().matmul(tensor_2.clone().exp());
    let grads = tensor_3.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    let tolerance = Tolerance::default();
    let expected = TensorData::from([[54.5991, 27.4746], [54.5991, 27.4746]]);
    grad_1
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);

    let expected = TensorData::from([[-5.4598e+01, -9.1188e-04], [2.9556e+01, 8.0342e+01]]);
    grad_2
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);
}
