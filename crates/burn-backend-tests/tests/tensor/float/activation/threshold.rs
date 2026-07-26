use super::*;
use burn_tensor::Tolerance;
use burn_tensor::{TensorData, activation};

#[test]
fn test_threshold_d2() {
    let tensor = TestTensor::<2>::from([[-1.0, 0.5, 1.0], [1.5, 2.0, -3.0]]);

    let output = activation::threshold(tensor, 1.0, 20.0);
    // threshold(x, 1, 20) = x if x > 1 else 20
    // row 0: all <= 1 -> 20; row 1: 1.5, 2.0 kept, -3.0 -> 20
    let expected = TensorData::from([[20.0, 20.0, 20.0], [1.5, 2.0, 20.0]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn test_threshold_zero_matches_relu() {
    let tensor = TestTensor::<1>::from([-2.0, 0.0, 3.0]);

    let output = activation::threshold(tensor, 0.0, 0.0);
    // threshold(x, 0, 0) = x if x > 0 else 0 = relu(x)
    let expected = TensorData::from([0.0, 0.0, 3.0]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}
