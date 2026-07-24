use super::*;
use burn_tensor::TensorData;

#[test]
fn test_any() {
    // test float tensor
    let tensor = TestTensor::<2>::from([[0.0, 0.0, 0.0], [1.0, -1.0, 0.0]]);
    let data_actual = tensor.any().into_data();
    let data_expected = TensorData::from([true]);
    data_expected.assert_eq(&data_actual, false);

    let tensor = TestTensor::<2>::from([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    let data_actual = tensor.any().into_data();
    let data_expected = TensorData::from([false]);
    data_expected.assert_eq(&data_actual, false);

    // test int tensor
    let tensor = TestTensorInt::<2>::from([[0, 0, 0], [1, -1, 0]]);
    let data_actual = tensor.any().into_data();
    let data_expected = TensorData::from([true]);
    data_expected.assert_eq(&data_actual, false);

    let tensor = TestTensorInt::<2>::from([[0, 0, 0], [0, 0, 0]]);
    let data_actual = tensor.any().into_data();
    let data_expected = TensorData::from([false]);
    data_expected.assert_eq(&data_actual, false);

    // test bool tensor
    let tensor = TestTensorBool::<2>::from([[false, false, false], [true, true, false]]);
    let data_actual = tensor.any().into_data();
    let data_expected = TensorData::from([true]);
    data_expected.assert_eq(&data_actual, false);

    let tensor = TestTensorBool::<2>::from([[false, false, false], [false, false, false]]);
    let data_actual = tensor.any().into_data();
    let data_expected = TensorData::from([false]);
    data_expected.assert_eq(&data_actual, false);
}

#[test]
fn test_any_dim() {
    let tensor = TestTensor::<2>::from([[0.0, 0.0, 0.0], [1.0, -1.0, 0.0]]);
    let data_actual = tensor.any_dim(1).into_data();
    let data_expected = TensorData::from([[false], [true]]);
    data_expected.assert_eq(&data_actual, false);

    // test int tensor
    let tensor = TestTensorInt::<2>::from([[0, 0, 0], [1, -1, 0]]);
    let data_actual = tensor.any_dim(1).into_data();
    let data_expected = TensorData::from([[false], [true]]);
    data_expected.assert_eq(&data_actual, false);

    // test bool tensor
    let tensor = TestTensorBool::<2>::from([[false, false, false], [true, true, false]]);
    let data_actual = tensor.any_dim(1).into_data();
    let data_expected = TensorData::from([[false], [true]]);
    data_expected.assert_eq(&data_actual, false);
}

/// Larger `any_dim` over a long, non-power-of-two axis so the reduction spans
/// multiple blocks/planes/cubes. Rows cover every merge case: all-zero,
/// all-nonzero, a single nonzero at the first/last/middle column, and an
/// all-nonzero row with a single zero. Checked for float, int and bool inputs.
#[test]
fn test_any_dim_large() {
    let device = Default::default();
    let (rows, cols) = (8usize, 513usize);
    let mask = build_logical_mask(rows, cols);

    let expected: Vec<bool> = (0..rows)
        .map(|r| (0..cols).any(|c| mask[r * cols + c]))
        .collect();
    let expected = TensorData::new(expected, [rows, 1]);

    let float = TestTensor::<2>::from_data(
        TensorData::new(mask_to_floats(&mask), [rows, cols]),
        &device,
    );
    expected.assert_eq(&float.any_dim(1).into_data(), false);

    let int =
        TestTensorInt::<2>::from_data(TensorData::new(mask_to_ints(&mask), [rows, cols]), &device);
    expected.assert_eq(&int.any_dim(1).into_data(), false);

    let boolean = TestTensorBool::<2>::from_data(TensorData::new(mask, [rows, cols]), &device);
    expected.assert_eq(&boolean.any_dim(1).into_data(), false);
}
