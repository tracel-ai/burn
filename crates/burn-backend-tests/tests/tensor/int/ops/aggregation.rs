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
