use crate::*;
use burn_tensor::{TensorData, Tolerance};

#[test]
fn should_diff_cos() {
    let data_1 = TensorData::from([[0.0, 1.0], [3.0, 4.0]]);
    let data_2 = TensorData::from([[6.0, 7.0], [9.0, 10.0]]);

    let device = Default::default();
    let tensor_1 = TestAutodiffTensor::<2>::from_data(data_1, &device).require_grad();
    let tensor_2 = TestAutodiffTensor::from_data(data_2, &device).require_grad();

    let tensor_3 = tensor_1.clone().matmul(tensor_2.clone().cos());
    let tensor_4 = tensor_3.matmul(tensor_2.clone());
    let grads = tensor_4.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    // Metal has less precise trigonometric functions
    let tolerance = Tolerance::default().set_half_precision_relative(1e-2);

    grad_1.to_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([[26.8063, -27.7870], [26.8063, -27.7870]]),
        tolerance,
    );

    grad_2.to_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([[9.222064, -39.123375], [-28.721354, 49.748356]]),
        tolerance,
    );
}
