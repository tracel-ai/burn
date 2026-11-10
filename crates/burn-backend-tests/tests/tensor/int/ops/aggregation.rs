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

#[test]
fn should_wrap_64_bit_sum_and_prod_on_overflow() {
    use burn_tensor::DType;
    let device = burn_tensor::Device::default();

    if device.supports_dtype(DType::I64) {
        let i64s = |values: Vec<i64>, shape: [usize; 2]| {
            TestTensorInt::<2>::from_data(TensorData::new(values, shape), (&device, DType::I64))
        };
        let sums = i64s(vec![i64::MAX, 1, 1, i64::MIN, -1, 0], [2, 3]);
        let products = i64s(vec![i64::MAX, 2, 1, i64::MIN, -1, 1], [2, 3]);

        sums.clone()
            .sum_dim(1)
            .into_data()
            .assert_eq(&TensorData::from([[i64::MIN + 1], [i64::MAX]]), false);
        sums.clone()
            .slice([0..1, 0..3])
            .sum()
            .into_data()
            .assert_eq(&TensorData::from([i64::MIN + 1]), false);
        sums.slice([0..1, 0..3])
            .mean_dim(1)
            .into_data()
            .assert_eq(&TensorData::from([[(i64::MIN + 1) / 3]]), false);
        products
            .clone()
            .prod_dim(1)
            .into_data()
            .assert_eq(&TensorData::from([[-2i64], [i64::MIN]]), false);
        products
            .slice([0..1, 0..3])
            .prod()
            .into_data()
            .assert_eq(&TensorData::from([-2i64]), false);
    }

    if device.supports_dtype(DType::U64) {
        let u64s = |values: Vec<u64>, shape: [usize; 2]| {
            TestTensorInt::<2>::from_data(TensorData::new(values, shape), (&device, DType::U64))
        };
        // Values on both sides of 2^63 also cover backends that compute u64 through i64.
        let sums = u64s(vec![u64::MAX, 1, 1, 1 << 63, 1 << 63, 0], [2, 3]);
        let products = u64s(vec![u64::MAX, 2, 1, 1 << 32, 1 << 32, 1], [2, 3]);

        sums.clone()
            .sum_dim(1)
            .into_data()
            .assert_eq(&TensorData::from([[1u64], [0]]), false);
        sums.slice([1..2, 0..3])
            .sum()
            .into_data()
            .assert_eq(&TensorData::from([0u64]), false);
        products
            .clone()
            .prod_dim(1)
            .into_data()
            .assert_eq(&TensorData::from([[u64::MAX - 1], [0]]), false);
        products
            .slice([1..2, 0..3])
            .prod()
            .into_data()
            .assert_eq(&TensorData::from([0u64]), false);
    }
}
