use super::*;
use burn_tensor::TensorData;

#[test]
fn test_should_mean_int() {
    let tensor = TestTensorInt::<2>::from([[2, 2, 2], [3, 4, 5]]);

    let output = tensor.mean();

    output.into_data().assert_eq(&TensorData::from([3]), false);
}

#[test]
fn test_should_mean_last_dim_int() {
    let tensor = TestTensorInt::<2>::from([[0, 1, 2], [3, 4, 5]]);

    let output = tensor.mean_dim(1);

    output
        .into_data()
        .assert_eq(&TensorData::from([[1], [4]]), false);
}

#[test]
fn test_should_sum_last_dim_int() {
    let tensor = TestTensorInt::<2>::from([[0, 1, 2], [3, 4, 5]]);

    let output = tensor.sum_dim(1);

    output
        .into_data()
        .assert_eq(&TensorData::from([[3], [12]]), false);
}

#[test]
fn test_should_sum_int() {
    let tensor = TestTensorInt::<2>::from([[0, 1, 2], [3, 4, 5]]);

    let output = tensor.sum();

    output.into_data().assert_eq(&TensorData::from([15]), false);
}

#[test]
#[ignore = "Not implemented for all backends yet"]
fn test_prod_int() {
    let tensor = TestTensorInt::<2>::from([[2, 1, 2], [3, 4, 5]]);
    let output = tensor.prod();

    output
        .into_data()
        .assert_eq(&TensorData::from([240]), false);

    let tensor_with_zero = TestTensorInt::<2>::from([[2, 0, 2], [3, 4, 5]]);
    let output = tensor_with_zero.prod();

    output.into_data().assert_eq(&TensorData::from([0]), false);
}

#[test]
#[ignore = "Not implemented for all backends yet"]
fn test_prod_dim_int() {
    let tensor = TestTensorInt::<2>::from([[2, 1, 2], [3, 4, 5]]);
    let output = tensor.prod_dim(1);
    output
        .into_data()
        .assert_eq(&TensorData::from([[4], [60]]), false);

    let tensor_with_zero = TestTensorInt::<2>::from([[2, 0, 2], [3, 4, 5]]);
    let output = tensor_with_zero.prod_dim(1);
    output
        .into_data()
        .assert_eq(&TensorData::from([[0], [60]]), false);

    // Negative Indexing.
    let tensor_with_zero = TestTensorInt::<2>::from([[2, 0, 2], [3, 4, 5]]);
    let output = tensor_with_zero.prod_dim(-1);
    output
        .into_data()
        .assert_eq(&TensorData::from([[0], [60]]), false);
}

#[test]
fn test_should_sum_empty_int() {
    let tensor = TestTensorInt::<1>::empty([0], &Default::default());

    let output = tensor.sum();

    output.into_data().assert_eq(&TensorData::from([0]), false);
}

#[test]
fn test_should_prod_empty_int() {
    let tensor = TestTensorInt::<1>::empty([0], &Default::default());

    let output = tensor.prod();

    output.into_data().assert_eq(&TensorData::from([1]), false);
}

// A float mean of nothing is `NaN`, but an integer has no such value, so this is rejected rather
// than reporting whatever a division by a zero count produces.
#[test]
#[should_panic]
fn test_mean_empty_int_should_panic() {
    let tensor = TestTensorInt::<1>::empty([0], &Default::default());

    let _ = tensor.mean().into_data();
}

#[test]
#[should_panic]
fn test_mean_dim_empty_axis_int_should_panic() {
    let tensor = TestTensorInt::<2>::empty([3, 0], &Default::default());

    let _ = tensor.mean_dim(1).into_data();
}

/// A tensor of `E` on the default device, or `None` where the device doesn't support `E`.
fn int_tensor<E: burn_tensor::Element, const D: usize>(
    values: Vec<E>,
    shape: [usize; D],
) -> Option<TestTensorInt<D>> {
    let device = burn_tensor::Device::default();
    device.supports_dtype(E::dtype()).then(|| {
        TestTensorInt::<D>::from_data(TensorData::new(values, shape), (&device, E::dtype()))
    })
}

#[test]
fn should_wrap_i64_sum_on_overflow() {
    let Some(tensor) = int_tensor(vec![i64::MAX, 1, 1], [3]) else {
        return;
    };
    tensor
        .sum()
        .into_data()
        .assert_eq(&TensorData::from([i64::MIN + 1]), false);
}

#[test]
fn should_wrap_i64_sum_dim_on_overflow() {
    let Some(tensor) = int_tensor(vec![i64::MAX, 1, 1, i64::MIN, -1, 0], [2, 3]) else {
        return;
    };
    tensor
        .sum_dim(1)
        .into_data()
        .assert_eq(&TensorData::from([[i64::MIN + 1], [i64::MAX]]), false);
}

#[test]
fn should_wrap_i64_mean_on_overflow() {
    let Some(tensor) = int_tensor(vec![i64::MAX, 1, 1], [3]) else {
        return;
    };
    tensor
        .mean()
        .into_data()
        .assert_eq(&TensorData::from([(i64::MIN + 1) / 3]), false);
}

#[test]
fn should_wrap_i64_mean_dim_on_overflow() {
    let Some(tensor) = int_tensor(vec![i64::MAX, 1, 1], [1, 3]) else {
        return;
    };
    tensor
        .mean_dim(1)
        .into_data()
        .assert_eq(&TensorData::from([[(i64::MIN + 1) / 3]]), false);
}

#[test]
fn should_wrap_i64_prod_on_overflow() {
    let Some(tensor) = int_tensor(vec![i64::MAX, 2, 1], [3]) else {
        return;
    };
    tensor
        .prod()
        .into_data()
        .assert_eq(&TensorData::from([-2i64]), false);
}

#[test]
fn should_wrap_i64_prod_dim_on_overflow() {
    let Some(tensor) = int_tensor(vec![i64::MAX, 2, 1, i64::MIN, -1, 1], [2, 3]) else {
        return;
    };
    tensor
        .prod_dim(1)
        .into_data()
        .assert_eq(&TensorData::from([[-2i64], [i64::MIN]]), false);
}

// The u64 values sit on both sides of 2^63, which also covers backends that compute u64 through
// i64.

#[test]
fn should_wrap_u64_sum_on_overflow() {
    let Some(tensor) = int_tensor(vec![1u64 << 63, 1 << 63, 0], [3]) else {
        return;
    };
    tensor
        .sum()
        .into_data()
        .assert_eq(&TensorData::from([0u64]), false);
}

#[test]
fn should_wrap_u64_sum_dim_on_overflow() {
    let Some(tensor) = int_tensor(vec![u64::MAX, 1, 1, 1 << 63, 1 << 63, 0], [2, 3]) else {
        return;
    };
    tensor
        .sum_dim(1)
        .into_data()
        .assert_eq(&TensorData::from([[1u64], [0]]), false);
}

#[test]
fn should_wrap_u64_prod_on_overflow() {
    let Some(tensor) = int_tensor(vec![1u64 << 32, 1 << 32, 1], [3]) else {
        return;
    };
    tensor
        .prod()
        .into_data()
        .assert_eq(&TensorData::from([0u64]), false);
}

#[test]
fn should_wrap_u64_prod_dim_on_overflow() {
    let Some(tensor) = int_tensor(vec![u64::MAX, 2, 1, 1 << 32, 1 << 32, 1], [2, 3]) else {
        return;
    };
    tensor
        .prod_dim(1)
        .into_data()
        .assert_eq(&TensorData::from([[u64::MAX - 1], [0]]), false);
}
