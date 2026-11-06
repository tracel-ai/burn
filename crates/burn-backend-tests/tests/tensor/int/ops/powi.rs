use super::*;
use burn_tensor::{DType, TensorData};

#[test]
fn should_support_powi_broadcast() {
    // [1, 3] broadcast against [2, 3]
    let tensor_1 = TestTensorInt::<2>::from([[1, 2, 3]]);
    let tensor_2 = TestTensorInt::from([[2, 2, 2], [3, 3, 3]]);

    let output = tensor_1.clone().powi(tensor_2);
    let expected = TensorData::from([[1, 4, 9], [1, 8, 27]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_support_powi_scalar() {
    let tensor = TestTensorInt::<1>::from([-2, 3, 4]);

    let output = tensor.powi_scalar(3);
    let expected = TensorData::from([-8, 27, 64]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_support_powi_negative_exponent() {
    // `1 / base^|exp|` truncates to 0 unless the base is 1 or -1.
    let base = TestTensorInt::<1>::from([3, -3, 0, 1, -1, -1]);
    let exp = TestTensorInt::from([-1, -3, -1, -2, -1, -2]);

    let output = base.powi(exp);
    let expected = TensorData::from([0, 0, 0, 1, -1, 1]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_support_powi_scalar_negative_exponent() {
    let tensor = TestTensorInt::<1>::from([3, -1]);

    let output = tensor.powi_scalar(-1);
    let expected = TensorData::from([0, -1]);
    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_support_powi_u32_exponent_above_i32_max() {
    let device = burn_tensor::Device::default();
    if !device.supports_dtype(DType::U32) {
        return;
    }
    let base = TestTensorInt::<1>::from_data([3u32], (&device, DType::U32));
    let exp = TestTensorInt::from_data([u32::MAX], (&device, DType::U32));

    base.powi(exp)
        .into_data()
        .assert_eq(&TensorData::from([3u32.wrapping_pow(u32::MAX)]), false);
}

#[test]
fn should_support_powi_u64_exponent_above_i64_max() {
    let device = burn_tensor::Device::default();
    if !device.supports_dtype(DType::U64) {
        return;
    }
    let base = TestTensorInt::<1>::from_data([3u64], (&device, DType::U64));
    let exp = TestTensorInt::from_data([u64::MAX], (&device, DType::U64));

    base.powi(exp)
        .into_data()
        .assert_eq(&TensorData::from([3u64.wrapping_pow(u32::MAX)]), false);
}

#[test]
fn should_support_powi_i64_exponents_outside_i32() {
    let device = burn_tensor::Device::default();
    if !device.supports_dtype(DType::I64) {
        return;
    }
    // Cover signed narrowing, negative parity, and the low 32 bits of positive exponents.
    let big = 1i64 << 32;
    let base = TestTensorInt::<1>::from_data([3i64, -1, -1, 2, 3, 3], (&device, DType::I64));
    let exp = TestTensorInt::from_data(
        [1 << 31, i64::MIN, i64::MIN + 1, i64::MIN, big, big + 1],
        (&device, DType::I64),
    );

    base.powi(exp).into_data().assert_eq(
        &TensorData::from([3i64.wrapping_pow(1 << 31), 1, -1, 0, 1, 3]),
        false,
    );
}

#[test]
#[should_panic(expected = "The provided tensors have incompatible shapes.")]
fn should_panic_powi_incompatible_shapes() {
    // Same rank, but [2, 2] vs [2, 3]: dimension 1 cannot broadcast.
    let tensor_1 = TestTensorInt::<2>::from([[1, 2], [3, 4]]);
    let tensor_2 = TestTensorInt::from([[2, 2, 2], [3, 3, 3]]);

    let output = tensor_1.powi(tensor_2);
    output.into_data();
}
