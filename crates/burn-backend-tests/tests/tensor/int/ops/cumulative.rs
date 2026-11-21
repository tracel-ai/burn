use super::*;
use burn_tensor::TensorData;

#[test]
fn test_cumsum_int_dim_0() {
    let tensor = TestTensorInt::<2>::from([[1, 2, 3], [4, 5, 6]]);

    let output = tensor.cumsum(0);

    output
        .into_data()
        .assert_eq(&TensorData::from([[1, 2, 3], [5, 7, 9]]), false);
}

#[test]
fn test_cumsum_int_dim_1() {
    let tensor = TestTensorInt::<2>::from([[1, 2, 3], [4, 5, 6]]);

    let output = tensor.cumsum(1);

    output
        .into_data()
        .assert_eq(&TensorData::from([[1, 3, 6], [4, 9, 15]]), false);
}

#[test]
fn test_cumsum_int_negative_dim() {
    let tensor = TestTensorInt::<2>::from([[1, 2, 3], [4, 5, 6]]);

    let output = tensor.cumsum(-1);

    output
        .into_data()
        .assert_eq(&TensorData::from([[1, 3, 6], [4, 9, 15]]), false);
}

#[test]
fn test_cumprod_int_dim_0() {
    let tensor = TestTensorInt::<2>::from([[1, 2, 3], [4, 5, 6]]);

    let output = tensor.cumprod(0);

    output
        .into_data()
        .assert_eq(&TensorData::from([[1, 2, 3], [4, 10, 18]]), false);
}

#[test]
fn test_cumprod_int_dim_1() {
    let tensor = TestTensorInt::<2>::from([[1, 2, 3], [4, 5, 6]]);

    let output = tensor.cumprod(1);

    output
        .into_data()
        .assert_eq(&TensorData::from([[1, 2, 6], [4, 20, 120]]), false);
}

#[test]
fn test_cumprod_int_negative_dim() {
    let tensor = TestTensorInt::<2>::from([[1, 2, 3], [4, 5, 6]]);

    let output = tensor.cumprod(-1);

    output
        .into_data()
        .assert_eq(&TensorData::from([[1, 2, 6], [4, 20, 120]]), false);
}

#[test]
fn test_cummin_int_dim_0() {
    let tensor = TestTensorInt::<2>::from([[3, 1, 4], [2, 5, 1]]);

    let output = tensor.cummin(0);

    output
        .into_data()
        .assert_eq(&TensorData::from([[3, 1, 4], [2, 1, 1]]), false);
}

#[test]
fn test_cummin_int_dim_1() {
    let tensor = TestTensorInt::<2>::from([[3, 1, 4], [2, 5, 1]]);

    let output = tensor.cummin(1);

    output
        .into_data()
        .assert_eq(&TensorData::from([[3, 1, 1], [2, 2, 1]]), false);
}

#[test]
fn test_cummin_int_negative_dim() {
    let tensor = TestTensorInt::<2>::from([[3, 1, 4], [2, 5, 1]]);

    let output = tensor.cummin(-1);

    output
        .into_data()
        .assert_eq(&TensorData::from([[3, 1, 1], [2, 2, 1]]), false);
}

#[test]
fn test_cummax_int_dim_0() {
    let tensor = TestTensorInt::<2>::from([[3, 1, 4], [1, 5, 2]]);

    let output = tensor.cummax(0);

    output
        .into_data()
        .assert_eq(&TensorData::from([[3, 1, 4], [3, 5, 4]]), false);
}

#[test]
fn test_cummax_int_dim_1() {
    let tensor = TestTensorInt::<2>::from([[3, 1, 4], [1, 5, 2]]);

    let output = tensor.cummax(1);

    output
        .into_data()
        .assert_eq(&TensorData::from([[3, 3, 4], [1, 5, 5]]), false);
}

#[test]
fn test_cummax_int_negative_dim() {
    let tensor = TestTensorInt::<2>::from([[3, 1, 4], [1, 5, 2]]);

    let output = tensor.cummax(-1);

    output
        .into_data()
        .assert_eq(&TensorData::from([[3, 3, 4], [1, 5, 5]]), false);
}

/// A tensor of `E` on the default device, or `None` where the device doesn't support `E`.
fn int_tensor<E: burn_tensor::Element>(values: [E; 3]) -> Option<TestTensorInt<1>> {
    let device = burn_tensor::Device::default();
    device.supports_dtype(E::dtype()).then(|| {
        TestTensorInt::<1>::from_data(TensorData::new(values.to_vec(), [3]), (&device, E::dtype()))
    })
}

#[test]
fn should_wrap_i32_cumsum_on_overflow() {
    let Some(tensor) = int_tensor([i32::MAX, 1, 1]) else {
        return;
    };
    tensor
        .cumsum(0)
        .into_data()
        .assert_eq(&TensorData::from([i32::MAX, i32::MIN, i32::MIN + 1]), false);
}

#[test]
fn should_wrap_i32_cumprod_on_overflow() {
    let Some(tensor) = int_tensor([i32::MAX, 2, 1]) else {
        return;
    };
    tensor
        .cumprod(0)
        .into_data()
        .assert_eq(&TensorData::from([i32::MAX, -2, -2]), false);
}

#[test]
fn should_wrap_i64_cumsum_on_overflow() {
    let Some(tensor) = int_tensor([i64::MAX, 1, 1]) else {
        return;
    };
    tensor
        .cumsum(0)
        .into_data()
        .assert_eq(&TensorData::from([i64::MAX, i64::MIN, i64::MIN + 1]), false);
}

#[test]
fn should_wrap_i64_cumprod_on_overflow() {
    let Some(tensor) = int_tensor([i64::MAX, 2, 1]) else {
        return;
    };
    tensor
        .cumprod(0)
        .into_data()
        .assert_eq(&TensorData::from([i64::MAX, -2, -2]), false);
}

#[test]
fn should_wrap_u64_cumsum_on_overflow() {
    let Some(tensor) = int_tensor([u64::MAX, 1, 1]) else {
        return;
    };
    tensor
        .cumsum(0)
        .into_data()
        .assert_eq(&TensorData::from([u64::MAX, 0, 1]), false);
}

#[test]
fn should_wrap_u64_cumprod_on_overflow() {
    let Some(tensor) = int_tensor([1u64 << 63, 2, 3]) else {
        return;
    };
    tensor
        .cumprod(0)
        .into_data()
        .assert_eq(&TensorData::from([1u64 << 63, 0, 0]), false);
}
