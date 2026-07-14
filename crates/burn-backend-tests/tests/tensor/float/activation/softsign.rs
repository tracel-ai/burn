use super::*;
use burn_tensor::Tolerance;
use burn_tensor::{TensorData, activation};

#[test]
fn test_softsign() {
    let tensor = TestTensor::<2>::from([[1.0, 7.0], [13.0, -3.0]]);

    let output = activation::softsign(tensor);
    let expected = TensorData::from([[0.5, 0.875], [0.928571, -0.75]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn test_softsign_zero() {
    let tensor = TestTensor::<1>::from([0.0]);

    let output = activation::softsign(tensor);
    let expected = TensorData::from([0.0]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}
