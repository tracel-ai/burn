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
    // Negative zero must map to +0.0, not -1.0. Guards the `x == 0.0` branch in
    // the `copysign`-based implementations: `copysign(1.0, -0.0)` is `-1.0`, so
    // dropping that branch silently changes the result for negative zero.
    let tensor = TestTensor::<2>::from([[-0.0, 0.0]]);

    let output = tensor.sign().into_data().convert::<f32>();
    let output = output.as_slice::<f32>().unwrap();

    // `assert_eq` on `TensorData` uses ordinary float equality, where `-0.0 == 0.0`,
    // so it would not catch a regression that returns negative zero here. Compare
    // the raw bit patterns instead to actually enforce the +0.0 guarantee.
    assert_eq!(output[0].to_bits(), 0.0f32.to_bits());
    assert_eq!(output[1].to_bits(), 0.0f32.to_bits());
}
