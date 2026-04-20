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

// Only run under the `flex` backend feature; other burn backends follow
// IEEE 754 min/max and drop NaN. Positive-gate form because the default
// CI build doesn't set identifying feature flags on burn-backend-tests.
// See issue #4814.
#[cfg(feature = "flex")]
#[test]
fn test_argmax_nan_propagation() {
    let tensor = TestTensor::<2>::from([[1.0, f32::NAN, 3.0]]);
    let output = tensor.argmax(1);

    output
        .into_data()
        .assert_eq(&TensorData::from([[1]]), false);
}
