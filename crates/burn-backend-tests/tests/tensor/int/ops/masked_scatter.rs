use super::*;
use burn_tensor::TensorData;

#[test]
fn test_masked_scatter_int_1d() {
    let tensor = TestTensorInt::<1>::from([0, 0, 0, 0, 0]);
    let mask = TestTensorBool::<1>::from([true, false, true, true, false]);
    let source = TestTensorInt::<1>::from([10, 20, 30]);

    let output = tensor.masked_scatter(mask, source);

    output
        .into_data()
        .assert_eq(&TensorData::from([10, 0, 20, 30, 0]), false);
}

#[test]
fn test_masked_scatter_int_2d_row_major() {
    let tensor = TestTensorInt::<2>::from([[0, 0, 0], [0, 0, 0]]);
    let mask = TestTensorBool::<2>::from([[true, false, false], [true, true, false]]);
    let source = TestTensorInt::<1>::from([1, 2, 3]);

    let output = tensor.masked_scatter(mask, source);

    output
        .into_data()
        .assert_eq(&TensorData::from([[1, 0, 0], [2, 3, 0]]), false);
}

#[test]
fn test_masked_scatter_no_true_positions() {
    let tensor = TestTensorInt::<1>::from([1, 2, 3]);
    let mask = TestTensorBool::<1>::from([false, false, false]);
    let source =
        TestTensorInt::<1>::from_data(TensorData::new(Vec::<i32>::new(), [0]), &Default::default());

    let output = tensor.masked_scatter(mask, source);

    output
        .into_data()
        .assert_eq(&TensorData::from([1, 2, 3]), false);
}

#[test]
#[should_panic]
fn test_masked_scatter_short_source_panics() {
    let tensor = TestTensorInt::<1>::from([0, 0, 0]);
    let mask = TestTensorBool::<1>::from([true, true, true]);
    let source = TestTensorInt::<1>::from([1, 2]);

    let _ = tensor.masked_scatter(mask, source);
}
