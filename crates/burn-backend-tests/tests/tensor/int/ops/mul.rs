use super::*;
use burn_tensor::TensorData;

#[test]
fn should_support_mul_ops_int() {
    let data_1 = TensorData::from([[0, 1, 2], [3, 4, 5]]);
    let data_2 = TensorData::from([[0, 1, 2], [3, 4, 5]]);
    let device = Default::default();
    let tensor_1 = TestTensorInt::<2>::from_data(data_1, &device);
    let tensor_2 = TestTensorInt::<2>::from_data(data_2, &device);

    let output = tensor_1 * tensor_2;
    let expected = TensorData::from([[0, 1, 4], [9, 16, 25]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn test_mul_broadcast_int() {
    let data_1 = TensorData::from([[0, 1, 2]]);
    let data_2 = TensorData::from([[3, 4, 5], [6, 7, 8]]);
    let device = Default::default();
    let tensor_1 = TestTensorInt::<2>::from_data(data_1, &device);
    let tensor_2 = TestTensorInt::<2>::from_data(data_2, &device);

    let output = tensor_1 * tensor_2;
    let expected = TensorData::from([[0, 4, 10], [0, 7, 16]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_support_mul_scalar_ops_int() {
    let data = TensorData::from([[0, 1, 2], [3, 4, 5]]);
    let scalar = 2;
    let tensor = TestTensorInt::<2>::from_data(data, &Default::default());

    let output = tensor * scalar;
    let expected = TensorData::from([[0, 2, 4], [6, 8, 10]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn test_int_mul_flipped_2d() {
    // [[1, 2], [3, 4]] axis 0 flipped -> [[3, 4], [1, 2]]
    // * [[10, 20], [30, 40]] = [[30, 80], [30, 80]]
    let a = TestTensorInt::<2>::from([[1, 2], [3, 4]]).flip([0]);
    let b = TestTensorInt::<2>::from([[10, 20], [30, 40]]);

    let output = a * b;

    output
        .into_data()
        .assert_eq(&TensorData::from([[30, 80], [30, 80]]), false);
}

#[test]
fn test_int_mul_flipped_both_axes() {
    // [[1, 2], [3, 4]] flipped on both axes -> [[4, 3], [2, 1]]
    // * [[5, 5], [5, 5]] = [[20, 15], [10, 5]]
    let a = TestTensorInt::<2>::from([[1, 2], [3, 4]]).flip([0, 1]);
    let b = TestTensorInt::<2>::from([[5, 5], [5, 5]]);

    let output = a * b;

    output
        .into_data()
        .assert_eq(&TensorData::from([[20, 15], [10, 5]]), false);
}

#[test]
fn should_wrap_mul_i64_on_overflow() {
    let device = burn_tensor::Device::default();
    if !device.supports_dtype(burn_tensor::DType::I64) {
        return;
    }
    let lhs =
        TestTensorInt::<1>::from_data([i64::MAX, i64::MIN, 3], (&device, burn_tensor::DType::I64));
    let rhs = TestTensorInt::<1>::from_data([2i64, -1, 2], (&device, burn_tensor::DType::I64));

    lhs.clone()
        .mul(rhs)
        .into_data()
        .assert_eq(&TensorData::from([-2i64, i64::MIN, 6]), false);
    lhs.mul_scalar(2)
        .into_data()
        .assert_eq(&TensorData::from([-2i64, 0, 6]), false);
}

#[test]
fn should_wrap_mul_u64_on_overflow() {
    let device = burn_tensor::Device::default();
    if !device.supports_dtype(burn_tensor::DType::U64) {
        return;
    }
    // Values on both sides of 2^63 also cover backends that compute u64 through i64.
    let lhs = TestTensorInt::<1>::from_data(
        [1u64 << 62, u64::MAX, 3],
        (&device, burn_tensor::DType::U64),
    );
    let rhs = TestTensorInt::<1>::from_data([2u64, 2, 2], (&device, burn_tensor::DType::U64));

    lhs.clone()
        .mul(rhs)
        .into_data()
        .assert_eq(&TensorData::from([1u64 << 63, u64::MAX - 1, 6]), false);
    lhs.mul_scalar(2)
        .into_data()
        .assert_eq(&TensorData::from([1u64 << 63, u64::MAX - 1, 6]), false);
}
