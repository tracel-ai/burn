use super::*;
use burn_tensor::TensorData;

#[test]
fn test_argmax_2d_dim0() {
    let tensor = TestTensor::<2>::from([[10.0, 11.0, 2.0], [3.0, 4.0, 5.0]]);

    let output = tensor.argmax(0);

    output
        .into_data()
        .assert_eq(&TensorData::from([[0, 0, 1]]), false);
}

#[test]
fn test_argmin_2d_dim0() {
    let tensor = TestTensor::<2>::from([[10.0, 11.0, 2.0], [30.0, 4.0, 5.0]]);

    let output = tensor.argmin(0);

    output
        .into_data()
        .assert_eq(&TensorData::from([[0, 1, 0]]), false);
}

#[test]
fn test_argmax_2d_dim1() {
    let tensor = TestTensor::<2>::from([[10.0, 11.0, 2.0], [3.0, 4.0, 5.0]]);

    let output = tensor.argmax(1);

    output
        .into_data()
        .assert_eq(&TensorData::from([[1], [2]]), false);
}

#[test]
fn test_argmin_2d_dim1() {
    let tensor = TestTensor::<2>::from([[10.0, 11.0, 2.0], [30.0, 4.0, 5.0]]);

    let output = tensor.argmin(1);

    output
        .into_data()
        .assert_eq(&TensorData::from([[2], [1]]), false);
}

#[test]
fn test_argmax_flipped() {
    // Flip [1, 5, 3, 2, 4] -> [4, 2, 3, 5, 1]; max is at index 3.
    let tensor = TestTensor::<1>::from([1.0, 5.0, 3.0, 2.0, 4.0]);
    let output = tensor.flip([0]).argmax(0);

    output.into_data().assert_eq(&TensorData::from([3]), false);
}

#[test]
fn test_argmax_2d_flipped() {
    // [[1, 5, 3], [6, 2, 4]] axis-1-flipped -> [[3, 5, 1], [4, 2, 6]]; argmax dim 1 -> [[1], [2]].
    let tensor = TestTensor::<2>::from([[1.0, 5.0, 3.0], [6.0, 2.0, 4.0]]);
    let output = tensor.flip([1]).argmax(1);

    output
        .into_data()
        .assert_eq(&TensorData::from([[1], [2]]), false);
}

#[test]
fn test_argmin_flipped() {
    // Flip [5, 1, 4, 2, 3] -> [3, 2, 4, 1, 5]; min is at index 3.
    let tensor = TestTensor::<1>::from([5.0, 1.0, 4.0, 2.0, 3.0]);
    let output = tensor.flip([0]).argmin(0);

    output.into_data().assert_eq(&TensorData::from([3]), false);
}

#[test]
fn test_argmax_permuted_4d() {
    // Regression: argmax on a permuted 4D tensor (was index OOB).
    let n = 2 * 3 * 4 * 5;
    let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let tensor =
        TestTensor::<4>::from_data(TensorData::new(data, [2, 3, 4, 5]), &Default::default());
    let permuted = tensor.permute([0, 2, 1, 3]);

    let result = permuted.clone().argmax(3);
    assert_eq!(result.dims(), [2, 4, 3, 1]);

    let result = permuted.argmax(2);
    assert_eq!(result.dims(), [2, 4, 1, 5]);
}

#[test]
fn test_argmin_permuted_4d() {
    let n = 2 * 3 * 4 * 5;
    let data: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
    let tensor =
        TestTensor::<4>::from_data(TensorData::new(data, [2, 3, 4, 5]), &Default::default());
    let permuted = tensor.permute([0, 2, 1, 3]);

    let result = permuted.argmin(3);
    assert_eq!(result.dims(), [2, 4, 3, 1]);
}

#[test]
fn test_argmax_4d_middle_dim() {
    // Regression (YOLOv8n): shape [1, 84, 80, 80], argmax dim=1.
    let n = 1 * 84 * 80 * 80;
    let data: Vec<f32> = (0..n).map(|i| (i % 84) as f32).collect();
    let tensor =
        TestTensor::<4>::from_data(TensorData::new(data, [1, 84, 80, 80]), &Default::default());

    let output = tensor.argmax(1);
    assert_eq!(output.dims(), [1, 1, 80, 80]);
}

#[test]
fn test_argmax_permuted_correctness() {
    // Data [2, 2, 3] permuted [0, 2, 1] -> [2, 3, 2]; argmax dim 2 should be all 1s.
    let data: Vec<f32> = (1..=12).map(|i| i as f32).collect();
    let tensor = TestTensor::<3>::from_data(TensorData::new(data, [2, 2, 3]), &Default::default());
    let output = tensor.permute([0, 2, 1]).argmax(2);

    output
        .into_data()
        .assert_eq(&TensorData::from([[[1], [1], [1]], [[1], [1], [1]]]), false);
}

// NaN-propagation tests below. Only run when the `flex` backend feature
// is active, because flex is the only burn backend that currently
// propagates NaN from argmax/argmin (matching PyTorch/NumPy/JAX/TF).
// ndarray and the cubecl backends follow IEEE 754 min/max and drop NaN.
// The positive-gate form (rather than excluding specific backends) is
// used because the default-feature CI build selects a backend
// transitively without setting any of its identifying feature flags on
// burn-backend-tests, so a negative gate would still run the test on a
// NaN-dropping backend. See issue #4814.
#[cfg(feature = "flex")]
#[test]
fn test_argmax_nan_propagation() {
    let tensor = TestTensor::<2>::from([[1.0, f32::NAN, 3.0]]);
    let output = tensor.argmax(1);

    output
        .into_data()
        .assert_eq(&TensorData::from([[1]]), false);
}

// First-NaN wins: when row[0] is NaN AND a later element is also NaN,
// argmax must return 0 (the earlier NaN index), not the later one.
#[cfg(feature = "flex")]
#[test]
fn test_argmax_nan_leading_with_trailing_nan() {
    let tensor = TestTensor::<2>::from([[f32::NAN, f32::NAN, 3.0]]);
    let output = tensor.argmax(1);
    output
        .into_data()
        .assert_eq(&TensorData::from([[0]]), false);
}

#[cfg(feature = "flex")]
#[test]
fn test_argmin_nan_leading_with_trailing_nan() {
    let tensor = TestTensor::<2>::from([[f32::NAN, f32::NAN, 3.0]]);
    let output = tensor.argmin(1);
    output
        .into_data()
        .assert_eq(&TensorData::from([[0]]), false);
}

// All-NaN row: argmax should return 0 (first NaN).
#[cfg(feature = "flex")]
#[test]
fn test_argmax_all_nan_row() {
    let tensor = TestTensor::<2>::from([[f32::NAN, f32::NAN, f32::NAN]]);
    let output = tensor.argmax(1);
    output
        .into_data()
        .assert_eq(&TensorData::from([[0]]), false);
}

// NaN propagation must also hold for non-last-dim argmax, which routes
// through a different kernel than the last-dim fast path.
#[cfg(feature = "flex")]
#[test]
fn test_argmax_nan_leading_non_last_dim() {
    // Column 0: [NaN, NaN, 3.0] -> argmax along dim 0 = 0.
    // Column 1: [1.0, 2.0, 4.0] -> argmax along dim 0 = 2.
    let tensor = TestTensor::<2>::from([[f32::NAN, 1.0], [f32::NAN, 2.0], [3.0, 4.0]]);
    let output = tensor.argmax(0);
    output
        .into_data()
        .assert_eq(&TensorData::from([[0, 2]]), false);
}

// max_dim_with_indices on a leading-NaN row should return (NaN, 0).
// Complements test_max_dim_with_indices_nan_propagation in maxmin.rs,
// which uses a single NaN in the middle of the row.
#[cfg(feature = "flex")]
#[test]
fn test_max_dim_with_indices_nan_leading() {
    let tensor = TestTensor::<2>::from([[f32::NAN, f32::NAN, 3.0]]);
    let (values, indices) = tensor.max_dim_with_indices(1);
    let vdata = values.into_data();
    let slice = vdata.as_slice::<FloatElem>().unwrap();
    assert!(slice[0].is_nan());
    indices
        .into_data()
        .assert_eq(&TensorData::from([[0]]), false);
}
