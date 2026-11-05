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
fn powi_i64_exponents_outside_i32_should_preserve_powers() {
    let device = burn_tensor::Device::default();
    if !device.supports_dtype(DType::I64) {
        return;
    }
    // Exercise mixed signs as well as vectors containing only negative or non-negative exponents.
    let base = [3i64, -1, 2, -1, 3, -1, -1, 0, 3, 3, -1, 1];
    // A plain cast to `i32` would turn `-2^32` into 0.
    let big = 1i64 << 32;
    let exp = [
        1 << 31,
        -big,
        -big,
        -big - 1,
        i64::MIN,
        i64::MIN,
        i64::MIN + 1,
        -1,
        1 << 31,
        big,
        big + 1,
        2 * big,
    ];

    let lhs = TestTensorInt::<1>::from_data(TensorData::from(base), (&device, DType::I64));
    let rhs = TestTensorInt::<1>::from_data(TensorData::from(exp), (&device, DType::I64));

    lhs.powi(rhs).into_data().assert_eq(
        &TensorData::from([
            3i64.wrapping_pow(1 << 31),
            1,
            0,
            -1,
            0,
            1,
            -1,
            0,
            3i64.wrapping_pow(1 << 31),
            1,
            -1,
            1,
        ]),
        false,
    );
}
