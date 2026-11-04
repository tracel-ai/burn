use super::*;
use burn_tensor::{DType, TensorData};

#[test]
fn powi_u32_exponents_above_i32_max_should_stay_powers() {
    let device = Default::default();
    let base = [3u32, 2, 5, 7];
    // Read as `i32`, the first three exponents would be negative.
    let exp = [3_000_000_001u32, 1 << 31, u32::MAX, 3];

    let lhs = TestTensorInt::<1>::from_data(TensorData::from(base), (&device, DType::U32));
    let rhs = TestTensorInt::<1>::from_data(TensorData::from(exp), (&device, DType::U32));

    let expected: [u32; 4] = core::array::from_fn(|i| base[i].wrapping_pow(exp[i]));
    lhs.powi(rhs)
        .into_data()
        .assert_eq(&TensorData::from(expected), false);
}

#[test]
fn powi_i64_exponents_outside_i32_should_keep_sign_and_parity() {
    let device = Default::default();
    let base = [-1i64, -1, 2, 3, -1, 1];
    // A plain cast to `i32` would turn `-2^32` into 0.
    let big = 1i64 << 32;
    let exp = [-big, -big - 1, -big, -(1 << 40), big + 1, 2 * big];

    let lhs = TestTensorInt::<1>::from_data(TensorData::from(base), (&device, DType::I64));
    let rhs = TestTensorInt::<1>::from_data(TensorData::from(exp), (&device, DType::I64));

    lhs.powi(rhs)
        .into_data()
        .assert_eq(&TensorData::from([1i64, -1, 0, 0, -1, 1]), false);
}
