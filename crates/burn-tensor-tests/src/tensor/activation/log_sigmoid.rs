use crate::*;
use burn_tensor::Tolerance;
use burn_tensor::{ElementConversion, TensorData, activation};

#[test]
fn test_log_sigmoid() {
    let tensor = TestTensor::<2>::from([[1.0, 7.0], [13.0, -3.0]]);

    let output = activation::log_sigmoid(tensor);
    let expected = TensorData::from([[-3.132617e-1, -9.114665e-4], [-2.260327e-6, -3.0485873]]);

    let tolerance = Tolerance::rel_abs(0.01, 0.0001);
    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);
}

#[test]
fn test_log_sigmoid_numerical_stability() {
    let tensor = TestTensor::<1>::from([300.0, -300.0]);

    let output = activation::log_sigmoid(tensor);

    // For large negative values, the previous implementation −log(1 + exp(−x)) would give -inf
    let expected = TensorData::from([0.0, -300.0]);
    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());

    let tensor = TestTensor::<1>::from([FloatElem::MAX, FloatElem::MIN]);
    let output = activation::log_sigmoid(tensor);
    let expected = TensorData::from([0.elem(), FloatElem::MIN]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}
