use super::*;
use burn_tensor::Tolerance;
use burn_tensor::{TensorData, activation};

#[test]
fn test_selu() {
    // selu(x) = gamma * x if x > 0, gamma * alpha * (exp(x) - 1) if x <= 0
    // alpha = 1.6733, gamma = 1.0507
    let tensor = TestTensor::<2>::from([[0.0, 1.0, -1.0], [2.0, -2.0, 0.5]]);

    let output = activation::selu(tensor);

    // Expected values computed from the formula:
    // selu(0.0)  = 1.0507 * 1.6733 * (exp(0) - 1) = 0.0
    // selu(1.0)  = 1.0507 * 1.0 = 1.0507
    // selu(-1.0) = 1.0507 * 1.6733 * (exp(-1) - 1) = 1.7581 * (0.3679 - 1) = -1.1113
    // selu(2.0)  = 1.0507 * 2.0 = 2.1014
    // selu(-2.0) = 1.0507 * 1.6733 * (exp(-2) - 1) = 1.7581 * (0.1353 - 1) = -1.5202
    // selu(0.5)  = 1.0507 * 0.5 = 0.5254
    let expected = TensorData::from([[0.0, 1.0507, -1.1113], [2.1014, -1.5202, 0.5254]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn test_selu_zero() {
    let tensor = TestTensor::<1>::from([0.0]);

    let output = activation::selu(tensor);
    let expected = TensorData::from([0.0]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}
