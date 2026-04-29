use super::*;
use burn_tensor::Tolerance;
use burn_tensor::{TensorData, activation};

#[test]
fn test_sigmoid() {
    let tensor = TestTensor::<2>::from([[1.0, 7.0], [13.0, -3.0]]);

    let output = activation::sigmoid(tensor);
    let expected = TensorData::from([[0.731059, 0.999089], [0.999998, 0.047426]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn test_sigmoid_d1() {
    let tensor = TestTensor::<1>::from([-10.0, 0.0, 10.0]);

    let output = activation::sigmoid(tensor);

    output.into_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([0.0, 0.5, 1.0]),
        Tolerance::absolute(1e-3),
    );
}

#[test]
fn test_sigmoid_overflow() {
    let tensor = TestTensor::<1>::from([FloatElem::MAX, FloatElem::MIN]);

    let output = activation::sigmoid(tensor);
    let expected = TensorData::from([1.0, 0.0]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}
