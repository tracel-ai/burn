use super::*;
use burn_tensor::TensorData;

#[test]
fn test_median_even() {
    let tensor = TestTensor::<2>::from_data(
        [[0.5, 1.8, 0.2, -2.0], [3.0, -4.0, 5.0, 0.0]],
        &Default::default(),
    );

    let median_actual_1 = tensor.clone().median(1);
    let median_expected_1 = TensorData::from([[0.2], [0.0]]).convert::<FloatElem>();
    median_actual_1
        .into_data()
        .assert_eq(&median_expected_1, false);

    let median_actual_0 = tensor.median(0);
    let median_expected_0 = TensorData::from([[0.5, -4.0, 0.2, -2.0]]).convert::<FloatElem>();
    median_actual_0
        .into_data()
        .assert_eq(&median_expected_0, false);
}

#[test]
fn test_median_odd() {
    let tensor = TestTensor::<2>::from_data(
        [
            [0.5, 1.8, 0.2, -2.0, 1.0],
            [3.0, -4.0, 5.0, 0.0, -1.0],
            [5.0, -5.0, 1.0, 3.0, -2.0],
        ],
        &Default::default(),
    );

    let median_actual_1 = tensor.clone().median(1);
    let median_expected_1 = TensorData::from([[0.5], [0.0], [1.0]]).convert::<FloatElem>();
    median_actual_1
        .into_data()
        .assert_eq(&median_expected_1, false);

    let median_actual_0 = tensor.median(0);
    let median_expected_0 = TensorData::from([[3.0, -4.0, 1.0, 0.0, -1.0]]).convert::<FloatElem>();
    median_actual_0
        .into_data()
        .assert_eq(&median_expected_0, false);
}

#[test]
fn test_median_with_indices() {
    let device = Default::default();
    let tensor = TestTensor::<1>::from_data([3.0, 1.0, 2.0], &device);
    // median = 2, original index = 2
    let (values, indices) = tensor.median_with_indices(0);
    values
        .into_data()
        .assert_eq(&TensorData::from([2.0]), false);
    indices
        .into_data()
        .assert_eq(&TensorData::from([2i64]), false);

    let tensor = TestTensor::<2>::from_data([[5.0, 1.0, 3.0], [2.0, 8.0, 4.0]], &device);
    // Along dim 1:
    // Row 0: median = 3, original index = 2
    // Row 1: median = 4, original index = 2
    let (values, indices) = tensor.median_with_indices(1);
    values
        .into_data()
        .assert_eq(&TensorData::from([[3.0], [4.0]]), false);
    indices
        .into_data()
        .assert_eq(&TensorData::from([[2i64], [2i64]]), false);
}

#[test]
fn test_median_all_elements() {
    let tensor = TestTensor::<2>::from_data(
        [
            [0.5, 1.8, 0.2, -2.0, 1.0],
            [3.0, -4.0, 5.0, 0.0, -1.0],
            [5.0, -5.0, 1.0, 3.0, -2.0],
        ],
        &Default::default(),
    );

    // Sorted: [-5, -4, -2, -2, -1, 0, 0.2, 0.5, 1, 1, 1.8, 3, 3, 5, 5]
    let dims = tensor.dims().len();
    let flattened_tensor: Tensor<_, 1> = tensor.flatten(0, dims - 1);
    let result = flattened_tensor.median(0);
    result
        .into_data()
        .assert_eq(&TensorData::from([0.5]), false);
}
