use super::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
fn should_support_round_ops() {
    let data = TensorData::from([[24.0423, 87.9478, 76.1838], [59.6929, 43.8169, 94.8826]]);
    let tensor = TestTensor::<2>::from_data(data, &Default::default());

    let output = tensor.round();
    let expected = TensorData::from([[24., 88., 76.], [60., 44., 95.]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_round_ties_even() {
    let data = TensorData::from([1.5, 2.5, 3.5, 4.5, 5.5, 6.5]);
    let tensor = TestTensor::<1>::from_data(data, &Default::default());

    let output = tensor.round();
    let expected = TensorData::from([2., 2., 4., 4., 6., 6.]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_support_round_to_decimals() {
    let data = TensorData::from([[24.0423, 87.9478, 76.1838], [59.6929, 43.8169, 94.8826]]);
    let tensor = TestTensor::<2>::from_data(data, &Default::default());

    let output = tensor.round_to(2);
    let expected = TensorData::from([[24.04, 87.95, 76.18], [59.69, 43.82, 94.88]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_support_round_to_negative_values() {
    let data = TensorData::from([-1.2346, -2.3456, -3.4567]);
    let tensor = TestTensor::<1>::from_data(data, &Default::default());

    let output = tensor.round_to(3);
    let expected = TensorData::from([-1.235, -2.346, -3.457]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn round_to_zero_matches_round() {
    let data = TensorData::from([-1.4, -0.6, 0.6, 1.4, 2.6, 3.6]);
    let tensor = TestTensor::<1>::from_data(data, &Default::default());

    tensor
        .clone()
        .round_to(0)
        .into_data()
        .assert_approx_eq::<FloatElem>(&tensor.round().into_data(), Tolerance::default());
}

#[test]
fn should_support_round_to_negative_decimals() {
    // Negative decimals round to the left of the decimal point, keeping the
    // round-half-to-even tie breaking of `round`.
    let data = TensorData::from([1234.56, -1234.56, 987.65, 1550.0, 1450.0, 0.0]);
    let tensor = TestTensor::<1>::from_data(data, &Default::default());

    let output = tensor.round_to(-2);
    let expected = TensorData::from([1200.0, -1200.0, 1000.0, 1600.0, 1400.0, 0.0]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
#[should_panic(expected = "round_to")]
fn round_to_rejects_unrepresentable_decimals() {
    // The scale factor is built as an f64, so decimals beyond what an f64 can
    // represent are rejected instead of silently doing nothing.
    let tensor = TestTensor::<1>::from([1.2345]);
    let _ = tensor.round_to(i32::MAX);
}
