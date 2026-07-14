use super::*;
use burn_tensor::Tolerance;
use burn_tensor::{TensorData, activation};

#[test]
fn test_elu() {
    let tensor = TestTensor::<2>::from([[1.0, 7.0], [13.0, -3.0]]);

    let output = activation::elu(tensor, 1.0);
    // elu(1, 1) = 1, elu(7, 1) = 7, elu(13, 1) = 13
    // elu(-3, 1) = 1 * (exp(-3) - 1) = -0.950213
    let expected = TensorData::from([[1.0, 7.0], [13.0, -0.950213]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn test_elu_alpha() {
    let tensor = TestTensor::<1>::from([0.0, -1.0, -2.0]);

    let output = activation::elu(tensor, 2.0);
    // elu(0, 2) = 2*(exp(0)-1) = 0
    // elu(-1, 2) = 2*(exp(-1)-1) = 2*(-0.632121) = -1.264241
    // elu(-2, 2) = 2*(exp(-2)-1) = 2*(-0.864665) = -1.729329
    let expected = TensorData::from([0.0, -1.264241, -1.729329]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}
