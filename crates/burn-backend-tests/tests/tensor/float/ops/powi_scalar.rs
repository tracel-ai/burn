use super::*;
use burn_tensor::{TensorData, Tolerance};

#[test]
fn powi_scalar_negative_exponent() {
    TestTensor::<1>::from([-2.0, -1.0, 0.5, 2.0])
        .powi_scalar(-3)
        .into_data()
        .assert_approx_eq::<FloatElem>(
            &TensorData::from([-0.125, -1.0, 8.0, 0.125]),
            Tolerance::default(),
        );
}

#[test]
fn powi_scalar_supports_i32_max_exponent() {
    // Converting the exponent to f32 would round this odd integer to an even one.
    let tensor = TestTensor::<1>::from([-1.0, 1.0, -1.0, 1.0]);
    let expected = TensorData::from([-1.0, 1.0, -1.0, 1.0]);
    tensor
        .clone()
        .powi_scalar(i32::MAX)
        .into_data()
        .assert_eq(&expected, false);
    tensor
        .powf_scalar(i32::MAX as f64)
        .into_data()
        .assert_eq(&expected, false);
}

#[test]
fn powi_scalar_supports_i32_min_exponent() {
    TestTensor::<1>::from([-1.0, 1.0, 0.5, 2.0])
        .powi_scalar(i32::MIN)
        .into_data()
        .assert_eq(&TensorData::from([1.0, 1.0, f32::INFINITY, 0.0]), false);
}

#[test]
fn powi_scalar_preserves_out_of_range_fallback() {
    // Saturating to i32::MAX would change this even exponent to an odd one.
    // Wrapping to i32::MIN would exchange the zero and infinity results.
    // Use f32 so the powf fallback's exponent stays finite.
    TestTensor::<1>::from([-1.0, 1.0, 0.5, 2.0])
        .cast(burn_tensor::DType::F32)
        .powi_scalar(i32::MAX as i64 + 1)
        .into_data()
        .assert_eq(&TensorData::from([1.0, 1.0, 0.0, f32::INFINITY]), false);
}

#[test]
fn powi_scalar_preserves_fractional_fallback() {
    TestTensor::<1>::from([4.0, 16.0])
        .powi_scalar(3.5)
        .into_data()
        .assert_approx_eq::<FloatElem>(&TensorData::from([128.0, 16384.0]), Tolerance::default());
}

#[test]
fn powi_scalar_strided_chain() {
    let tensor = TestTensor::<2>::from([[-2.0, -1.0, 0.5], [2.0, 1.0, -0.5]]).transpose();
    let output = tensor.powi_scalar(3).mul_scalar(2.0).add_scalar(1.0);
    output.into_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([[-15.0, 17.0], [-1.0, 3.0], [1.25, 0.75]]),
        Tolerance::default(),
    );
}
