use super::*;
use burn_tensor::{TensorData, Tolerance, activation};

#[test]
fn should_diff_softplus() {
    let device = AutodiffDevice::new();
    let tensor = TestTensor::<1>::from_data([-1.0, 0.0, 2.0], &device).require_grad();

    let output = activation::softplus(tensor.clone(), 1.0, 20.0).sum();
    let grads = output.backward();

    // The derivative of softplus is `sigmoid(beta * x)`.
    let expected = TensorData::from([0.268941, 0.5, 0.880797]);
    tensor
        .grad(&grads)
        .unwrap()
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_diff_softplus_saturated() {
    let device = AutodiffDevice::new();
    // A naive `log(exp(x) + 1)` forward saturates to `inf` here, which yields a `NaN`
    // gradient and poisons the rest of the backward pass.
    let tensor = TestTensor::<1>::from_data([-100.0, 0.0, 100.0], &device).require_grad();

    let output = activation::softplus(tensor.clone(), 1.0, 20.0).sum();
    let grads = output.backward();

    // `sigmoid` saturates to 0 and 1 without ever becoming `NaN`.
    let expected = TensorData::from([0.0, 0.5, 1.0]);
    tensor
        .grad(&grads)
        .unwrap()
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}
