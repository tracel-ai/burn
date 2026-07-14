use super::*;
use burn_tensor::Tolerance;
use burn_tensor::{TensorData, activation};

#[test]
fn test_hard_sigmoid() {
    let tensor = TestTensor::<2>::from([[1.0, 7.0], [13.0, -3.0]]);

    let output = activation::hard_sigmoid(tensor, 0.2, 0.5);
    let expected = TensorData::from([[0.7, 1.0], [1.0, 0.0]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn test_hard_sigmoid_overflow() {
    let tensor = TestTensor::<1>::from([FloatElem::MAX, FloatElem::MIN]);

    let output = activation::hard_sigmoid(tensor, 0.2, 0.5);
    let expected = TensorData::from([1.0, 0.0]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}
