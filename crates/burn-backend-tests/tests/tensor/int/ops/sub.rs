use super::*;
use burn_tensor::TensorData;

#[test]
fn should_support_sub_ops_int() {
    let data_1 = TensorData::from([[0, 1, 2], [3, 4, 5]]);
    let data_2 = TensorData::from([[6, 7, 8], [9, 10, 11]]);
    let device = Default::default();
    let tensor_1 = TestTensorInt::<2>::from_data(data_1, &device);
    let tensor_2 = TestTensorInt::<2>::from_data(data_2, &device);

    let output = tensor_1 - tensor_2;
    let expected = TensorData::from([[-6, -6, -6], [-6, -6, -6]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn test_sub_broadcast_int() {
    let data_1 = TensorData::from([[0, 1, 2]]);
    let data_2 = TensorData::from([[3, 4, 5], [6, 7, 8]]);
    let device = Default::default();
    let tensor_1 = TestTensorInt::<2>::from_data(data_1, &device);
    let tensor_2 = TestTensorInt::<2>::from_data(data_2, &device);

    let output = tensor_1 - tensor_2;
    let expected = TensorData::from([[-3, -3, -3], [-6, -6, -6]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_support_sub_scalar_ops_int() {
    let data = TensorData::from([[0, 1, 2], [3, 4, 5]]);
    let scalar = 2;
    let tensor = TestTensorInt::<2>::from_data(data, &Default::default());

    let output = tensor - scalar;
    let expected = TensorData::from([[-2, -1, 0], [1, 2, 3]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn test_int_sub_flipped() {
    // [10, 20, 30, 40] flipped - [1, 2, 3, 4] = [39, 28, 17, 6]
    let a = TestTensorInt::<1>::from([10, 20, 30, 40]).flip([0]);
    let b = TestTensorInt::<1>::from([1, 2, 3, 4]);

    let output = a - b;

    output
        .into_data()
        .assert_eq(&TensorData::from([39, 28, 17, 6]), false);
}

#[test]
fn should_wrap_sub_i64_on_overflow() {
    let device = burn_tensor::Device::default();
    if !device.supports_dtype(burn_tensor::DType::I64) {
        return;
    }
    let lhs =
        TestTensorInt::<1>::from_data([i64::MIN, i64::MAX, 3], (&device, burn_tensor::DType::I64));
    let rhs = TestTensorInt::<1>::from_data([1i64, -1, 2], (&device, burn_tensor::DType::I64));

    lhs.clone()
        .sub(rhs)
        .into_data()
        .assert_eq(&TensorData::from([i64::MAX, i64::MIN, 1]), false);
    lhs.sub_scalar(1)
        .into_data()
        .assert_eq(&TensorData::from([i64::MAX, i64::MAX - 1, 2]), false);
}

#[test]
fn should_wrap_sub_u64_on_overflow() {
    let device = burn_tensor::Device::default();
    if !device.supports_dtype(burn_tensor::DType::U64) {
        return;
    }
    // Values on both sides of 2^63 also cover backends that compute u64 through i64.
    let lhs = TestTensorInt::<1>::from_data([1u64 << 63, 0, 3], (&device, burn_tensor::DType::U64));
    let rhs = TestTensorInt::<1>::from_data([1u64, 1, 2], (&device, burn_tensor::DType::U64));

    lhs.clone()
        .sub(rhs)
        .into_data()
        .assert_eq(&TensorData::from([i64::MAX as u64, u64::MAX, 1]), false);
    lhs.sub_scalar(1)
        .into_data()
        .assert_eq(&TensorData::from([i64::MAX as u64, u64::MAX, 2]), false);
}
