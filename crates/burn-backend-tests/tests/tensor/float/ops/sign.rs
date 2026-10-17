use super::*;
use burn_tensor::TensorData;

#[test]
fn should_support_sign_ops_float() {
    let tensor = TestTensor::<2>::from([[-0.2, -1.0, 2.0], [3.0, 0.0, -5.0]]);

    let output = tensor.sign();
    let expected = TensorData::from([[-1.0, -1.0, 1.0], [1.0, 0.0, -1.0]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_support_sign_ops_float_negative_zero() {
    // Both signed zeros must map to zero. This guards against implementations
    // that use `copysign(1.0, x)` without handling zero, returning -1.0 for -0.0.
    let tensor = TestTensor::<2>::from([[-0.0, 0.0]]);

    let output = tensor.sign().into_data().convert::<f32>();
    let output = output.as_slice::<f32>().unwrap();

    // The contract does not specify the output zero's sign bit, so ordinary
    // float equality accepts either signed zero while still rejecting +/-1.0.
    assert_eq!(output[0], 0.0);
    assert_eq!(output[1], 0.0);
}

#[test]
fn should_support_sign_ops_float_nan() {
    // `sign(NaN)` must be `0` on every backend. Covers both NaN sign bits, since a
    // sign-bit-keyed implementation can pass on one and fail on the other.
    let tensor = TestTensor::<1>::from([f32::NAN, -f32::NAN]);

    let output = tensor.sign().into_data().convert::<f32>();
    let output = output.as_slice::<f32>().unwrap();

    assert_eq!(output[0], 0.0);
    assert_eq!(output[1], 0.0);
}
