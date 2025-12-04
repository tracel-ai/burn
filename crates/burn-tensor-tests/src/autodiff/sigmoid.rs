use crate::*;
use burn_tensor::Tolerance;
use burn_tensor::{TensorData, activation};

#[test]
fn should_diff_sigmoid() {
    let data = TensorData::from([0.8762]);

    let device = Default::default();
    let tensor_1 = TestAutodiffTensor::<1>::from_data(data, &device).require_grad();
    let tensor_2 = activation::sigmoid(tensor_1.clone());
    let grads = tensor_2.backward();

    let grad = tensor_1.grad(&grads).unwrap();

    let expected = TensorData::from([0.207549]);
    grad.to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn small_neg_val_should_not_cause_grad_overflow() {
    let data = TensorData::from([-90.0]);

    let device = Default::default();
    let tensor_1 = TestAutodiffTensor::<1>::from_data(data, &device).require_grad();
    let tensor_2 = activation::sigmoid(tensor_1.clone());
    let grads = tensor_2.backward();

    let grad = tensor_1.grad(&grads).unwrap();

    let expected = TensorData::from([0.0]);
    grad.to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}
