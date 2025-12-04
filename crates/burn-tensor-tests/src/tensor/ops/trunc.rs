use super::*;
use burn_tensor::Tolerance;
use burn_tensor::{ElementConversion, TensorData};

#[test]
fn should_support_trunc_ops() {
    let data = TensorData::from([[2.3, -1.7, 0.5], [-0.5, 3.9, -4.2]]);
    let tensor = TestTensor::<2>::from_data(data, &Default::default());

    let output = tensor.trunc();
    let expected = TensorData::from([[2.0, -1.0, 0.0], [0.0, 3.0, -4.0]]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_truncate_positive_values_like_floor() {
    let data = TensorData::from([1.7, 2.9, 3.1, 4.5]);
    let tensor = TestTensor::<1>::from_data(data, &Default::default());

    let output = tensor.trunc();
    let expected = TensorData::from([1.0, 2.0, 3.0, 4.0]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_truncate_negative_values_like_ceil() {
    let data = TensorData::from([-1.7, -2.9, -3.1, -4.5]);
    let tensor = TestTensor::<1>::from_data(data, &Default::default());

    let output = tensor.trunc();
    let expected = TensorData::from([-1.0, -2.0, -3.0, -4.0]);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_handle_special_cases() {
    // Test special IEEE 754 cases
    let data = TensorData::from([0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN]);
    let tensor = TestTensor::<1>::from_data(data, &Default::default());

    let output = tensor.trunc();
    let values = output.into_data().as_slice::<FloatElem>().unwrap().to_vec();

    // Check positive zero
    assert_eq!(values[0], 0.0f32.elem::<FloatElem>());
    assert!(values[0].is_sign_positive());

    // Check negative zero is preserved
    assert_eq!(values[1], 0.0f32.elem::<FloatElem>());
    assert!(values[1].is_sign_negative());

    // Check infinity is preserved
    assert!(values[2].is_infinite() && values[2].is_sign_positive());
    assert!(values[3].is_infinite() && values[3].is_sign_negative());

    // Check NaN is preserved
    assert!(values[4].is_nan());
}
