use crate::*;
use burn_tensor::Tolerance;
use burn_tensor::{TensorData, activation};

#[test]
fn should_diff_log_sigmoid() {
    let data = TensorData::from([[0.8762, -0.1423], [-300., 200.]]);

    let device = Default::default();
    let tensor_1 = TestAutodiffTensor::<2>::from_data(data, &device).require_grad();
    let tensor_2 = activation::log_sigmoid(tensor_1.clone());
    let grads = tensor_2.backward();

    let grad = tensor_1.grad(&grads).unwrap();

    let expected = TensorData::from([[0.293966, 0.535515], [1.000000, 0.000000]]);
    grad.to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}
