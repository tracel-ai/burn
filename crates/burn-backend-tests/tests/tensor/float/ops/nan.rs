use super::*;

#[test]
fn is_nan() {
    let no_nan = TestTensor::<2>::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let no_nan_expected = TestTensorBool::<2>::from([[false, false, false], [false, false, false]]);

    let with_nan = TestTensor::<2>::from([[0.0, f32::NAN, 2.0], [f32::NAN, 4.0, 5.0]]);
    let with_nan_expected = TestTensorBool::<2>::from([[false, true, false], [true, false, false]]);

    assert_eq!(no_nan_expected.into_data(), no_nan.is_nan().into_data());

    assert_eq!(with_nan_expected.into_data(), with_nan.is_nan().into_data());
}

#[test]
fn contains_nan() {
    let no_nan = TestTensor::<2>::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    assert!(!no_nan.contains_nan().into_scalar::<bool>());

    let with_nan = TestTensor::<2>::from([[0.0, f32::NAN, 2.0], [3.0, 4.0, 5.0]]);
    assert!(with_nan.contains_nan().into_scalar::<bool>());

    // Regression guard: a finite tensor must never be reported as containing NaN.
    // The previous `sum().is_nan()` implementation could overflow the reduction
    // on narrow-exponent floats (e.g. f16) and turn a finite tensor into a NaN sum,
    // producing a false positive. Here, Inf + (-Inf) = NaN.
    let device = Default::default();
    let pos = TestTensor::<1>::ones([2048], &device) * 60000.0;
    let neg = TestTensor::<1>::ones([2048], &device) * -60000.0;
    // Large positive and large negative numbers
    let large_finite = TestTensor::<1>::cat(vec![pos, neg], 0);
    assert!(!large_finite.contains_nan().into_scalar::<bool>());
}
