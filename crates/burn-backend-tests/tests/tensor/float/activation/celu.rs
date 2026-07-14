use super::*;
use burn_tensor::Tolerance;
use burn_tensor::{TensorData, activation};

#[test]
fn test_celu_d2() {
    let tensor = TestTensor::<2>::from([[1.0, 7.0], [-3.0, 0.5]]);

    let output = activation::celu(tensor, 1.0);
    // celu(1, 1) = 1
    // celu(7, 1) = 7
    // celu(-3, 1) = 1 * (exp(-3) - 1) = -0.950213
    // celu(0.5, 1) = 0.5
    let expected = TensorData::from([[1.0, 7.0], [-0.950213, 0.5]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn test_celu_with_alpha() {
    let tensor = TestTensor::<1>::from([0.0, -1.0, -2.0]);

    let output = activation::celu(tensor, 2.0);
    // celu(0, 2) = 0
    // celu(-1, 2) = 2 * (exp(-0.5) - 1) = -0.786939
    // celu(-2, 2) = 2 * (exp(-1) - 1) = -1.264241
    let expected = TensorData::from([0.0, -0.786939, -1.264241]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}
