use super::*;
use burn_tensor::{Shape, TensorData};

#[test]
fn test_mask_select_1d() {
    let tensor = TestTensor::<1>::from([1.0, 2.0, 3.0, 4.0, 5.0]);
    let mask = TestTensorBool::<1>::from([false, true, false, true, true]);

    let output = tensor.mask_select(mask);

    output
        .into_data()
        .assert_eq(&TensorData::from([2.0, 4.0, 5.0]), false);
}

#[test]
fn test_mask_select_2d() {
    let tensor = TestTensor::<2>::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let mask = TestTensorBool::<2>::from([[true, false, true], [false, true, false]]);

    let output = tensor.mask_select(mask);

    output
        .into_data()
        .assert_eq(&TensorData::from([1.0, 3.0, 5.0]), false);
}

#[test]
fn test_mask_select_3d() {
    // Flattened order across three dimensions: flat values are 1..=8, the mask keeps 1, 4, 5, 6.
    let tensor = TestTensor::<3>::from([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]);
    let mask = TestTensorBool::<3>::from([
        [[true, false], [false, true]],
        [[true, true], [false, false]],
    ]);

    let output = tensor.mask_select(mask);

    output
        .into_data()
        .assert_eq(&TensorData::from([1.0, 4.0, 5.0, 6.0]), false);
}

#[test]
fn test_mask_select_single_element() {
    // A single selected element exercises the `squeeze_dim(1)` path: `argwhere` returns shape
    // `[1, 1]`, which must squeeze to `[1]` (a global `squeeze` would collapse it to rank 0).
    let tensor = TestTensor::<2>::from([[1.0, 2.0], [3.0, 4.0]]);
    let mask = TestTensorBool::<2>::from([[false, false], [true, false]]);

    let output = tensor.mask_select(mask);

    output
        .into_data()
        .assert_eq(&TensorData::from([3.0]), false);
}

#[test]
fn test_mask_select_empty() {
    // An all-`false` mask selects nothing and yields an empty 1D tensor (no special-casing needed).
    let tensor = TestTensor::<1>::from([1.0, 2.0, 3.0]);
    let mask = TestTensorBool::<1>::from([false, false, false]);

    let output = tensor.mask_select(mask);

    assert_eq!(output.shape(), Shape::new([0]));
    // Read the data back to force execution on lazy backends (shape alone is metadata-only).
    let out: Vec<FloatElem> = output.into_data().to_vec().unwrap();
    assert!(out.is_empty());
}

#[test]
fn test_mask_select_transposed_tensor() {
    // [[1, 2], [3, 4]] transposed -> [[1, 3], [2, 4]]; mask [[T, F], [F, T]] selects 1 and 4.
    // Selection must follow the logical (flattened) order of the view, not the memory layout.
    let tensor = TestTensor::<2>::from([[1.0, 2.0], [3.0, 4.0]]).transpose();
    let mask = TestTensorBool::<2>::from([[true, false], [false, true]]);

    let output = tensor.mask_select(mask);

    output
        .into_data()
        .assert_eq(&TensorData::from([1.0, 4.0]), false);
}

#[test]
fn test_mask_select_both_flipped() {
    // tensor [1, 2, 3, 4] flipped -> [4, 3, 2, 1]; mask [T, F, T, F] flipped -> [F, T, F, T]
    // selects positions 1 and 3 of the flipped tensor: [3, 1].
    let tensor = TestTensor::<1>::from([1.0, 2.0, 3.0, 4.0]).flip([0]);
    let mask = TestTensorBool::<1>::from([true, false, true, false]).flip([0]);

    let output = tensor.mask_select(mask);

    output
        .into_data()
        .assert_eq(&TensorData::from([3.0, 1.0]), false);
}

#[test]
fn test_mask_select_narrowed_tensor() {
    // [1, 2, 3, 4, 5, 6] narrowed to [2, 3, 4, 5]; mask [T, F, F, T] selects [2, 5].
    let tensor = TestTensor::<1>::from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).narrow(0, 1, 4);
    let mask = TestTensorBool::<1>::from([true, false, false, true]);

    let output = tensor.mask_select(mask);

    output
        .into_data()
        .assert_eq(&TensorData::from([2.0, 5.0]), false);
}

#[test]
#[should_panic]
fn test_mask_select_shape_mismatch() {
    // The mask must have the same shape as the tensor: the flat indices of a differently-shaped
    // mask would not correspond to the tensor's positions.
    let tensor = TestTensor::<1>::from([1.0, 2.0, 3.0]);
    let mask = TestTensorBool::<1>::from([true, false]);

    let _output = tensor.mask_select(mask);
}
