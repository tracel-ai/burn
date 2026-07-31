use super::*;
use burn_tensor::TensorData;

#[test]
fn test_max_dim_with_indices_2d_int() {
    let tensor = TestTensorInt::<2>::from([[0, 1, 2], [3, 4, 5]]);

    let (values, indices) = tensor.clone().max_dim_with_indices(0);
    values
        .into_data()
        .assert_eq(&TensorData::from([[3, 4, 5]]), false);
    indices
        .into_data()
        .assert_eq(&TensorData::from([[1, 1, 1]]), false);

    let (values, indices) = tensor.max_dim_with_indices(1);
    values
        .into_data()
        .assert_eq(&TensorData::from([[2], [5]]), false);
    indices
        .into_data()
        .assert_eq(&TensorData::from([[2], [2]]), false);
}

#[test]
fn test_min_dim_with_indices_2d_int() {
    let tensor = TestTensorInt::<2>::from([[0, 1, 2], [3, 4, 5]]);

    let (values, indices) = tensor.clone().min_dim_with_indices(0);
    values
        .into_data()
        .assert_eq(&TensorData::from([[0, 1, 2]]), false);
    indices
        .into_data()
        .assert_eq(&TensorData::from([[0, 0, 0]]), false);

    let (values, indices) = tensor.min_dim_with_indices(1);
    values
        .into_data()
        .assert_eq(&TensorData::from([[0], [3]]), false);
    indices
        .into_data()
        .assert_eq(&TensorData::from([[0], [0]]), false);
}

// Ties resolve to the lowest index, which is what the two-pass argmax-then-gather
// default did and what the fused single-pass kernels have to keep doing.
#[test]
fn test_max_min_dim_with_indices_ties_int() {
    let tensor = TestTensorInt::<2>::from([[1, 3, 3], [2, 2, 0]]);

    let (values, indices) = tensor.clone().max_dim_with_indices(1);
    values
        .into_data()
        .assert_eq(&TensorData::from([[3], [2]]), false);
    indices
        .into_data()
        .assert_eq(&TensorData::from([[1], [0]]), false);

    let (values, indices) = tensor.min_dim_with_indices(1);
    values
        .into_data()
        .assert_eq(&TensorData::from([[1], [0]]), false);
    indices
        .into_data()
        .assert_eq(&TensorData::from([[0], [2]]), false);
}

#[test]
fn test_max_min_dim_with_indices_3d_int() {
    let tensor = TestTensorInt::<3>::from([[[1, 4, 7], [2, 5, 6]], [[3, 0, 9], [8, 2, 7]]]);

    let (values, indices) = tensor.clone().max_dim_with_indices(2);
    values
        .into_data()
        .assert_eq(&TensorData::from([[[7], [6]], [[9], [8]]]), false);
    indices
        .into_data()
        .assert_eq(&TensorData::from([[[2], [2]], [[2], [0]]]), false);

    let (values, indices) = tensor.min_dim_with_indices(2);
    values
        .into_data()
        .assert_eq(&TensorData::from([[[1], [2]], [[0], [2]]]), false);
    indices
        .into_data()
        .assert_eq(&TensorData::from([[[0], [0]], [[1], [1]]]), false);
}
