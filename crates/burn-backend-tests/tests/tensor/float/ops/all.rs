use super::*;
use burn_tensor::TensorData;

#[test]
fn test_all() {
    let tensor = TestTensor::<2>::from([[0.0, 1.0, 0.0], [1.0, -1.0, 1.0]]);
    let data_actual = tensor.all().into_data();
    let data_expected = TensorData::from([false]);
    data_expected.assert_eq(&data_actual, false);
}

#[test]
fn test_all_dim() {
    let tensor = TestTensor::<2>::from([[0.0, 1.0, 0.0], [1.0, -1.0, 1.0]]);
    let data_actual = tensor.all_dim(1).into_data();
    let data_expected = TensorData::from([[false], [true]]);
    data_expected.assert_eq(&data_actual, false);
}

/// Larger `all_dim` over a long, non-power-of-two axis so the reduction spans
/// multiple blocks/planes/cubes. Same mask as the large `any_dim` test; here a
/// row reduces to `true` only when every element is nonzero (so the single-zero
/// row 7 must reduce to `false`). Checked for float, int and bool inputs.
#[test]
fn test_all_dim_large() {
    let device = Default::default();
    let (rows, cols) = (8usize, 513usize);
    let mask = build_logical_mask(rows, cols);

    let expected: Vec<bool> = (0..rows)
        .map(|r| (0..cols).all(|c| mask[r * cols + c]))
        .collect();
    let expected = TensorData::new(expected, [rows, 1]);

    let float = TestTensor::<2>::from_data(
        TensorData::new(mask_to_floats(&mask), [rows, cols]),
        &device,
    );
    expected.assert_eq(&float.all_dim(1).into_data(), false);

    let int =
        TestTensorInt::<2>::from_data(TensorData::new(mask_to_ints(&mask), [rows, cols]), &device);
    expected.assert_eq(&int.all_dim(1).into_data(), false);

    let boolean = TestTensorBool::<2>::from_data(TensorData::new(mask, [rows, cols]), &device);
    expected.assert_eq(&boolean.all_dim(1).into_data(), false);
}
