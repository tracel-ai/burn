use super::TestTensorInt;
use burn_tensor::{DType, Device, TensorData};

#[test]
fn should_wrap_i64_sum_on_overflow() {
    let tensor = TestTensorInt::<1>::from_data([i64::MAX, 1, 1], (&Device::default(), DType::I64));

    tensor
        .sum()
        .into_data()
        .assert_eq(&TensorData::from([i64::MIN + 1]), false);
}

#[test]
fn should_wrap_i64_sum_dim_on_overflow() {
    let tensor = TestTensorInt::<2>::from_data(
        [[i64::MAX, 1, 1], [i64::MIN, -1, 0]],
        (&Device::default(), DType::I64),
    );

    tensor
        .sum_dim(1)
        .into_data()
        .assert_eq(&TensorData::from([[i64::MIN + 1], [i64::MAX]]), false);
}

#[test]
fn should_wrap_i64_mean_on_overflow() {
    let tensor = TestTensorInt::<1>::from_data([i64::MAX, 1, 1], (&Device::default(), DType::I64));

    tensor
        .mean()
        .into_data()
        .assert_eq(&TensorData::from([(i64::MIN + 1) / 3]), false);
}

#[test]
fn should_wrap_i64_mean_dim_on_overflow() {
    let tensor =
        TestTensorInt::<2>::from_data([[i64::MAX, 1, 1]], (&Device::default(), DType::I64));

    tensor
        .mean_dim(1)
        .into_data()
        .assert_eq(&TensorData::from([[(i64::MIN + 1) / 3]]), false);
}

#[test]
fn should_wrap_i64_prod_on_overflow() {
    let tensor = TestTensorInt::<1>::from_data([i64::MAX, 2, 1], (&Device::default(), DType::I64));

    tensor
        .prod()
        .into_data()
        .assert_eq(&TensorData::from([-2i64]), false);
}

#[test]
fn should_wrap_i64_prod_dim_on_overflow() {
    let tensor = TestTensorInt::<2>::from_data(
        [[i64::MAX, 2, 1], [i64::MIN, -1, 1]],
        (&Device::default(), DType::I64),
    );

    tensor
        .prod_dim(1)
        .into_data()
        .assert_eq(&TensorData::from([[-2i64], [i64::MIN]]), false);
}

// The u64 values sit on both sides of 2^63, which also covers backends that compute u64 through
// i64.

#[test]
fn should_wrap_u64_sum_on_overflow() {
    let tensor =
        TestTensorInt::<1>::from_data([1u64 << 63, 1 << 63, 0], (&Device::default(), DType::U64));

    tensor
        .sum()
        .into_data()
        .assert_eq(&TensorData::from([0u64]), false);
}

#[test]
fn should_wrap_u64_sum_dim_on_overflow() {
    let tensor = TestTensorInt::<2>::from_data(
        [[u64::MAX, 1, 1], [1 << 63, 1 << 63, 0]],
        (&Device::default(), DType::U64),
    );

    tensor
        .sum_dim(1)
        .into_data()
        .assert_eq(&TensorData::from([[1u64], [0]]), false);
}

#[test]
fn should_wrap_u64_prod_on_overflow() {
    let tensor = TestTensorInt::<1>::from_data(
        [1u64 << 32, (1u64 << 32) + 1],
        (&Device::default(), DType::U64),
    );

    tensor
        .prod()
        .into_data()
        .assert_eq(&TensorData::from([1u64 << 32]), false);
}

#[test]
fn should_wrap_u64_prod_dim_on_overflow() {
    let tensor = TestTensorInt::<2>::from_data(
        [[u64::MAX, 2, 1], [1 << 32, 1 << 32, 1]],
        (&Device::default(), DType::U64),
    );

    tensor
        .prod_dim(1)
        .into_data()
        .assert_eq(&TensorData::from([[u64::MAX - 1], [0]]), false);
}

#[test]
fn should_wrap_i64_cumsum_on_overflow() {
    let tensor = TestTensorInt::<1>::from_data([i64::MAX, 1, 1], (&Device::default(), DType::I64));

    tensor
        .cumsum(0)
        .into_data()
        .assert_eq(&TensorData::from([i64::MAX, i64::MIN, i64::MIN + 1]), false);
}

#[test]
fn should_wrap_i64_cumprod_on_overflow() {
    let tensor = TestTensorInt::<1>::from_data([i64::MAX, 2, 1], (&Device::default(), DType::I64));

    tensor
        .cumprod(0)
        .into_data()
        .assert_eq(&TensorData::from([i64::MAX, -2, -2]), false);
}

#[test]
fn should_wrap_u64_cumsum_on_overflow() {
    let tensor = TestTensorInt::<1>::from_data([u64::MAX, 1, 1], (&Device::default(), DType::U64));

    tensor
        .cumsum(0)
        .into_data()
        .assert_eq(&TensorData::from([u64::MAX, 0, 1]), false);
}

#[test]
fn should_wrap_u64_cumprod_on_overflow() {
    let tensor =
        TestTensorInt::<1>::from_data([1u64 << 63, 2, 3], (&Device::default(), DType::U64));

    tensor
        .cumprod(0)
        .into_data()
        .assert_eq(&TensorData::from([1u64 << 63, 0, 0]), false);
}
