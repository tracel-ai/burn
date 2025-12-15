use super::*;
use alloc::vec::Vec;
use burn_tensor::{Shape, TensorData};

#[test]
fn test_argwhere_1d() {
    let tensor = TestTensorBool::<1>::from([false, true, false, true, true]);
    let output = tensor.argwhere();

    output
        .into_data()
        .assert_eq(&TensorData::from([[1], [3], [4]]), false);
}

#[test]
fn test_argwhere_2d() {
    let tensor = TestTensorBool::<2>::from([[false, false], [false, true], [true, true]]);
    let output = tensor.argwhere();

    output
        .into_data()
        .assert_eq(&TensorData::from([[1, 1], [2, 0], [2, 1]]), false);
}

#[test]
fn test_argwhere_3d() {
    let tensor = TestTensorBool::<3>::from([
        [[false, false, false], [false, true, false]],
        [[true, false, true], [true, true, false]],
    ]);
    let output = tensor.argwhere();

    output.into_data().assert_eq(
        &TensorData::from([[0, 1, 1], [1, 0, 0], [1, 0, 2], [1, 1, 0], [1, 1, 1]]),
        false,
    );
}

#[test]
fn test_nonzero_1d() {
    let tensor = TestTensorBool::<1>::from([false, true, false, true, true]);
    let data_actual = tensor
        .nonzero()
        .into_iter()
        .map(|t| t.into_data())
        .collect::<Vec<_>>();

    assert_eq!(data_actual.len(), 1);
    data_actual[0].assert_eq(&TensorData::from([1, 3, 4]), false);
}

#[test]
fn test_nonzero_2d() {
    // 2-D tensor
    let tensor = TestTensorBool::<2>::from([[false, false], [false, true], [true, true]]);
    let data_actual = tensor
        .nonzero()
        .into_iter()
        .map(|t| t.into_data())
        .collect::<Vec<_>>();
    let data_expected = [TensorData::from([1, 2, 2]), TensorData::from([1, 0, 1])];

    assert_eq!(data_actual.len(), 2);
    for (idx, actual) in data_actual.iter().enumerate() {
        actual.assert_eq(&data_expected[idx], false)
    }
}

#[test]
fn test_nonzero_3d() {
    // 3-D tensor
    let tensor = TestTensorBool::<3>::from([
        [[false, false, false], [false, true, false]],
        [[true, false, true], [true, true, false]],
    ]);
    let data_actual = tensor
        .nonzero()
        .into_iter()
        .map(|t| t.into_data())
        .collect::<Vec<_>>();
    let data_expected = [
        TensorData::from([0, 1, 1, 1, 1]),
        TensorData::from([1, 0, 0, 1, 1]),
        TensorData::from([1, 0, 2, 0, 1]),
    ];

    assert_eq!(data_actual.len(), 3);
    for (idx, actual) in data_actual.iter().enumerate() {
        actual.assert_eq(&data_expected[idx], false)
    }
}

#[test]
fn test_nonzero_empty() {
    let tensor = TestTensorBool::<1>::from([false, false, false, false, false]);
    let output = tensor.nonzero();

    assert_eq!(output.len(), 0);
}

#[test]
fn test_argwhere_empty() {
    let tensor = TestTensorBool::<1>::from([false, false, false, false, false]);
    let output = tensor.argwhere();

    assert_eq!(output.shape(), Shape::new([0, 1]));
}
