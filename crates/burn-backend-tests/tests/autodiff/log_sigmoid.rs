use super::*;
use burn_tensor::Tolerance;
use burn_tensor::{TensorData, activation};

#[test]
fn should_diff_log_sigmoid() {
    let data = TensorData::from([[0.8762, -0.1423], [-300., 200.]]);

    let device = AutodiffDevice::new();
    let tensor_1 = TestTensor::<2>::from_data(data, &device).require_grad();
    // A non-uniform weight makes the incoming gradient differ per element, so
    // the backward pass must multiply by it: sigmoid(-x) * w.
    let weight = TestTensor::<2>::from_data([[2.0, -3.0], [0.5, 4.0]], &device);
    let tensor_2 = activation::log_sigmoid(tensor_1.clone()).mul(weight);
    let grads = tensor_2.backward();

    let grad = tensor_1.grad(&grads).unwrap();

    let expected = TensorData::from([[0.587932, -1.606545], [0.500000, 0.000000]]);
    grad.to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}
