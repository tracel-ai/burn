use super::*;
use burn_tensor::Tolerance;
use burn_tensor::{TensorData, activation};

#[test]
fn test_quiet_softmax_d2() {
    let tensor = TestTensor::<2>::from([[1.0, 7.0], [13.0, -3.0]]);

    let output = activation::quiet_softmax(tensor, 1);
    let expected = TensorData::from([[2.47e-03, 9.975e-01], [1.0, 1.1254e-07]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn test_quiet_softmax_negative_dim() {
    let tensor = TestTensor::<2>::from([[1.0, 7.0], [13.0, -3.0]]);

    let output = activation::quiet_softmax(tensor, -1);
    let expected = TensorData::from([[2.47e-03, 9.975e-01], [1.0, 1.1254e-07]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn test_quiet_softmax_all_negative_infinity_last_dim() {
    let tensor = TestTensor::<2>::from([
        [f32::NEG_INFINITY, f32::NEG_INFINITY],
        [f32::NEG_INFINITY, 0.0],
    ]);

    let output = activation::quiet_softmax(tensor, 1);
    let expected = TensorData::from([[0.0, 0.0], [0.0, 0.5]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn test_quiet_softmax_all_negative_infinity_first_dim() {
    let tensor = TestTensor::<2>::from([
        [f32::NEG_INFINITY, f32::NEG_INFINITY],
        [f32::NEG_INFINITY, 0.0],
    ]);

    let output = activation::quiet_softmax(tensor, 0);
    let expected = TensorData::from([[0.0, 0.0], [0.0, 0.5]]);

    output.into_data().assert_eq(&expected, false);
}
