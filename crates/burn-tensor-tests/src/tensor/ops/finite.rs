use super::*;

#[test]
fn is_finite() {
    let all_finite = TestTensor::<2>::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
    let all_finite_expected = TestTensorBool::<2>::from([[true, true, true], [true, true, true]]);

    let with_inf_nan = TestTensor::<2>::from([
        [0.0, f32::INFINITY, f32::NAN],
        [f32::NEG_INFINITY, f32::NAN, 5.0],
    ]);
    let with_inf_nan_expected =
        TestTensorBool::<2>::from([[true, false, false], [false, false, true]]);

    all_finite_expected
        .into_data()
        .assert_eq(&all_finite.is_finite().into_data(), false);

    with_inf_nan
        .is_finite()
        .into_data()
        .assert_eq(&with_inf_nan_expected.into_data(), false);
}
