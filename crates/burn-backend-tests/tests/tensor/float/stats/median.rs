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
fn test_median_nan_propagation() {
    // With NaN sorted last, the middle element of the first row is 3.0.
    let tensor = TestTensor::<2>::from([[3.0, f32::NAN, 1.0, f32::NAN], [4.0, 1.0, 3.0, 2.0]]);

    let values = tensor.median(1).into_data().convert::<f32>();
    let values = values.as_slice::<f32>().unwrap();

    assert!(values[0].is_nan());
    assert_eq!(values[1], 2.0);
}

#[test]
fn test_median_with_indices_nan_propagation() {
    let tensor = TestTensor::<2>::from([[3.0, f32::NAN, 1.0, f32::NAN], [4.0, 1.0, 3.0, 2.0]]);

    let (values, indices) = tensor.median_with_indices(1);
    let values = values.into_data().convert::<f32>();
    let values = values.as_slice::<f32>().unwrap();

    assert!(values[0].is_nan());
    assert_eq!(values[1], 2.0);
    // The first NaN, as for `max_dim_with_indices`.
    indices
        .into_data()
        .assert_eq(&TensorData::from([[1], [3]]), false);
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
    let flattened_tensor: Tensor<1> = tensor.flatten(0, dims - 1);
    let result = flattened_tensor.median(0);
    result
        .into_data()
        .assert_eq(&TensorData::from([0.5]), false);
}

#[test]
fn test_median_negative_dim() {
    let tensor =
        TestTensor::<2>::from_data([[5.0, 1.0, 3.0], [2.0, 8.0, 4.0]], &Default::default());

    let median = tensor.clone().median(1).into_data();
    tensor
        .clone()
        .median(-1)
        .into_data()
        .assert_eq(&median, false);

    let (values, indices) = tensor.clone().median_with_indices(1);
    let (values_negative, indices_negative) = tensor.median_with_indices(-1);
    values_negative
        .into_data()
        .assert_eq(&values.into_data(), false);
    indices_negative
        .into_data()
        .assert_eq(&indices.into_data(), false);
}
