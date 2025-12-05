use super::*;
use burn_tensor::Tolerance;
use burn_tensor::{ElementConversion, TensorData};

#[allow(unused_imports)] // f16
use num_traits::Float;

#[test]
fn should_support_fmod_ops() {
    let dividend = TensorData::from([[5.3, -5.3], [7.5, -7.5]]);
    let divisor = TensorData::from([[2.0, 2.0], [3.0, 3.0]]);

    let dividend_tensor = TestTensor::<2>::from_data(dividend, &Default::default());
    let divisor_tensor = TestTensor::<2>::from_data(divisor, &Default::default());

    let output = dividend_tensor.fmod(divisor_tensor);
    let expected = TensorData::from([[1.3, -1.3], [1.5, -1.5]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_support_fmod_scalar() {
    let data = TensorData::from([5.3, -5.3, 7.5, -7.5]);
    let tensor = TestTensor::<1>::from_data(data, &Default::default());

    let output = tensor.fmod_scalar(2.0);
    let expected = TensorData::from([1.3, -1.3, 1.5, -1.5]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_handle_positive_dividend_positive_divisor() {
    let dividend = TensorData::from([10.0, 7.5, 3.8, 1.2]);
    let divisor = TensorData::from([3.0, 2.0, 1.5, 0.7]);

    let dividend_tensor = TestTensor::<1>::from_data(dividend, &Default::default());
    let divisor_tensor = TestTensor::<1>::from_data(divisor, &Default::default());

    let output = dividend_tensor.fmod(divisor_tensor);
    let expected = TensorData::from([1.0, 1.5, 0.8, 0.5]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_handle_negative_dividend() {
    let dividend = TensorData::from([-10.0, -7.5, -3.8, -1.2]);
    let divisor = TensorData::from([3.0, 2.0, 1.5, 0.7]);

    let dividend_tensor = TestTensor::<1>::from_data(dividend, &Default::default());
    let divisor_tensor = TestTensor::<1>::from_data(divisor, &Default::default());

    let output = dividend_tensor.fmod(divisor_tensor);
    let expected = TensorData::from([-1.0, -1.5, -0.8, -0.5]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_handle_mixed_signs() {
    let dividend = TensorData::from([5.3, -5.3, 5.3, -5.3]);
    let divisor = TensorData::from([2.0, 2.0, -2.0, -2.0]);

    let dividend_tensor = TestTensor::<1>::from_data(dividend, &Default::default());
    let divisor_tensor = TestTensor::<1>::from_data(divisor, &Default::default());

    let output = dividend_tensor.fmod(divisor_tensor);
    // fmod result has same sign as dividend
    let expected = TensorData::from([1.3, -1.3, 1.3, -1.3]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_handle_infinity_dividend() {
    // If x is ±∞ and y is not NaN, NaN is returned
    let dividend = TensorData::from([
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ]);
    let divisor = TensorData::from([2.0, 3.0, -2.0, -3.0]);

    let dividend_tensor = TestTensor::<1>::from_data(dividend, &Default::default());
    let divisor_tensor = TestTensor::<1>::from_data(divisor, &Default::default());

    let output = dividend_tensor.fmod(divisor_tensor);
    let data = output.into_data();
    let values = data.as_slice::<FloatElem>().unwrap();

    // All results should be NaN
    assert!(values[0].is_nan(), "fmod(inf, 2.0) should be NaN");
    assert!(values[1].is_nan(), "fmod(-inf, 3.0) should be NaN");
    assert!(values[2].is_nan(), "fmod(inf, -2.0) should be NaN");
    assert!(values[3].is_nan(), "fmod(-inf, -3.0) should be NaN");
}

#[test]
fn should_handle_zero_divisor() {
    // If y is ±0 and x is not NaN, NaN should be returned
    let dividend = TensorData::from([5.3, -5.3, 0.0, 1.0]);
    let divisor = TensorData::from([0.0, -0.0, 0.0, -0.0]);

    let dividend_tensor = TestTensor::<1>::from_data(dividend, &Default::default());
    let divisor_tensor = TestTensor::<1>::from_data(divisor, &Default::default());

    let output = dividend_tensor.fmod(divisor_tensor);
    let data = output.into_data();
    let values = data.as_slice::<FloatElem>().unwrap();

    // All results should be NaN
    assert!(values[0].is_nan(), "fmod(5.3, 0.0) should be NaN");
    assert!(values[1].is_nan(), "fmod(-5.3, -0.0) should be NaN");
    assert!(values[2].is_nan(), "fmod(0.0, 0.0) should be NaN");
    assert!(values[3].is_nan(), "fmod(1.0, -0.0) should be NaN");
}

#[test]
fn should_handle_infinity_divisor() {
    // If y is ±∞ and x is finite, x is returned
    let dividend = TensorData::from([5.3, -5.3, 0.0, -0.0]);
    let divisor = TensorData::from([
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ]);

    let dividend_tensor = TestTensor::<1>::from_data(dividend, &Default::default());
    let divisor_tensor = TestTensor::<1>::from_data(divisor, &Default::default());

    let output = dividend_tensor.fmod(divisor_tensor);
    let expected = TensorData::from([5.3, -5.3, 0.0, -0.0]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_handle_nan_arguments() {
    // If either argument is NaN, NaN is returned
    let dividend = TensorData::from([f32::NAN, 5.3, f32::NAN, 0.0]);
    let divisor = TensorData::from([2.0, f32::NAN, f32::NAN, 3.0]);

    let dividend_tensor = TestTensor::<1>::from_data(dividend, &Default::default());
    let divisor_tensor = TestTensor::<1>::from_data(divisor, &Default::default());

    let output = dividend_tensor.fmod(divisor_tensor);
    let data = output.into_data();
    let values = data.as_slice::<FloatElem>().unwrap();

    assert!(values[0].is_nan(), "fmod(NaN, 2.0) should be NaN");
    assert!(values[1].is_nan(), "fmod(5.3, NaN) should be NaN");
    assert!(values[2].is_nan(), "fmod(NaN, NaN) should be NaN");
    assert!(!values[3].is_nan(), "fmod(0.0, 3.0) should be 0.0");
}

#[test]
fn should_handle_negative_zero() {
    // If x is -0 and y is greater than zero, either +0 or -0 may be returned
    let dividend = TensorData::from([-0.0_f32]);
    let divisor = TensorData::from([2.0_f32]);

    let dividend_tensor = TestTensor::<1>::from_data(dividend, &Default::default());
    let divisor_tensor = TestTensor::<1>::from_data(divisor, &Default::default());

    let output = dividend_tensor.fmod(divisor_tensor);
    let data = output.into_data();
    let values = data.as_slice::<FloatElem>().unwrap();

    // Result should be zero (either +0 or -0 is acceptable)
    assert_eq!(
        values[0],
        0.0f32.elem::<FloatElem>(),
        "fmod(-0, 2.0) should be zero"
    );
}

#[test]
fn should_support_fmod_broadcasting_2d() {
    // Test broadcasting: 1x2 with 3x2
    let dividend = TensorData::from([[5.3, -5.3]]); // Shape: 1x2
    let divisor = TensorData::from([[2.0, 2.0], [3.0, 3.0], [1.5, 1.5]]); // Shape: 3x2

    let dividend_tensor = TestTensor::<2>::from_data(dividend, &Default::default());
    let divisor_tensor = TestTensor::<2>::from_data(divisor, &Default::default());

    let output = dividend_tensor.fmod(divisor_tensor);
    let expected = TensorData::from([
        [1.3, -1.3], // 5.3 % 2.0, -5.3 % 2.0
        [2.3, -2.3], // 5.3 % 3.0, -5.3 % 3.0
        [0.8, -0.8], // 5.3 % 1.5, -5.3 % 1.5
    ]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_support_fmod_broadcasting_3d() {
    // Test broadcasting: 1x1x3 with 2x1x3
    let dividend = TensorData::from([[[5.0, -7.0, 8.0]]]); // Shape: 1x1x3
    let divisor = TensorData::from([[[3.0, 3.0, 3.0]], [[4.0, 4.0, 4.0]]]); // Shape: 2x1x3

    let dividend_tensor = TestTensor::<3>::from_data(dividend, &Default::default());
    let divisor_tensor = TestTensor::<3>::from_data(divisor, &Default::default());

    let output = dividend_tensor.fmod(divisor_tensor);
    let expected = TensorData::from([
        [[2.0, -1.0, 2.0]], // 5.0 % 3.0, -7.0 % 3.0, 8.0 % 3.0
        [[1.0, -3.0, 0.0]], // 5.0 % 4.0, -7.0 % 4.0, 8.0 % 4.0
    ]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_support_fmod_scalar_broadcasting() {
    // Test scalar operation with different shapes
    let data = TensorData::from([[5.3, -5.3, 7.5], [-7.5, 10.0, -10.0]]);
    let tensor = TestTensor::<2>::from_data(data, &Default::default());

    let output = tensor.fmod_scalar(3.0);
    let expected = TensorData::from([[2.3, -2.3, 1.5], [-1.5, 1.0, -1.0]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_handle_edge_case_values() {
    // Test various edge cases
    let dividend = TensorData::from([0.0, -0.0, 1e-10, -1e-10, 10.0, -10.0]);
    let divisor = TensorData::from([1.0, 1.0, 1.0, 1.0, 3.0, 3.0]);

    let dividend_tensor = TestTensor::<1>::from_data(dividend, &Default::default());
    let divisor_tensor = TestTensor::<1>::from_data(divisor, &Default::default());

    let output = dividend_tensor.fmod(divisor_tensor);
    let expected = TensorData::from([0.0, -0.0, 1e-10, -1e-10, 1.0, -1.0]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_handle_special_scalar_cases() {
    // Test scalar operations with special values
    let data = TensorData::from([5.3, -5.3, 0.0, -0.0]);
    let tensor = TestTensor::<1>::from_data(data, &Default::default());

    // Test with infinity divisor
    let output_inf = tensor.clone().fmod_scalar(f32::INFINITY);
    let expected_inf = TensorData::from([5.3, -5.3, 0.0, -0.0]);
    output_inf
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected_inf, Tolerance::default());

    // Test with very small divisor
    // Doesn't work if the test divisor is subnormal
    if FloatElem::MIN_POSITIVE > 1e-5f32.elem::<FloatElem>() {
        return;
    }

    let output_small = tensor.clone().fmod_scalar(1e-5);
    let data = output_small.into_data();
    let values = data.as_slice::<FloatElem>().unwrap();

    // let expected = TensorData::from([0.0, 0.0, 0.0, 0.0]);

    // Results should be very small remainders
    assert!(values[0].abs() < 1e-5f32.elem::<FloatElem>());
    assert!(values[1].abs() < 1e-5f32.elem::<FloatElem>());
    assert_eq!(values[2], 0.0f32.elem::<FloatElem>());
    assert_eq!(values[3], 0.0f32.elem::<FloatElem>());
}
